"""显式、只读的科研数据核验；不修改统计/标签/缓存，不作为普通训练启动门槛。"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def audit_provenance(dataset):
    """读历史包装记录并复核其载荷；旧验收字段只作为记录，不推断当前运行授权。"""
    records = []
    for path in sorted((dataset / "provenance").glob("*.json")):
        value = json.loads(path.read_text())
        fields = {key: value[key] for key in (
            "status", "training_enabled", "source_success_certified", "visual_contact_acceptance",
            "duplicate_raw_groups", "unique_raw_sources", "validation", "rendering", "rgb",
        ) if key in value}
        if not fields:
            continue
        record = dict(file=str(path.relative_to(dataset)), sha256=sha256(path), fields=fields)
        if path.name == "package.json":
            failures = []
            artifacts = value.get("artifacts", {})
            for name, expected in artifacts.items():
                target = dataset / name
                if (not target.resolve().is_relative_to(dataset.resolve()) or not target.is_file()
                    or target.stat().st_size != expected["bytes"]
                    or sha256(target) != expected["sha256"]):
                    failures.append(name)
            record.update(artifacts_checked=len(artifacts), artifacts_mismatched=failures)
        records.append(record)
    return records


def compare_cache_rgb(dataset, cache_root):
    """每条缓存轨迹独立重解码首帧、历史边界、尾帧及固定随机帧，逐像素/PTS 比较。"""
    import numpy as np
    import torch
    from torchcodec.decoders import VideoDecoder
    from rhsci.data.raw_input_cache import read_manifest

    torch.set_num_threads(1)
    directory, cache, digest = read_manifest(cache_root, dataset)
    count = 0
    for episode, record in cache["episodes"].items():
        length = record["frames"]
        query = [length - 1, 0, min(6, length - 1), 0]
        query += torch.randperm(length, generator=torch.Generator().manual_seed(
            2026 + int(episode)))[:16].tolist()
        decoder = VideoDecoder(str(dataset / record["source"]), device="cpu",
                               seek_mode="approximate", num_ffmpeg_threads=1)
        frames = decoder.get_frames_at(indices=query)
        pixels = np.load(directory / record["pixels"], mmap_mode="r", allow_pickle=False)
        pts = np.load(directory / record["pts"], mmap_mode="r", allow_pickle=False)
        np.testing.assert_array_equal(pixels[query], frames.data.numpy())
        np.testing.assert_array_equal(pts[query], frames.pts_seconds.numpy())
        count += len(query)
    return dict(manifest_sha256=digest, episodes=len(cache["episodes"]), frame_queries=count,
                pixels_and_pts_exact=True, scope="sampled decode in every cached episode")


def probe_video(pair):
    path, length = pair
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "stream=nb_frames,avg_frame_rate,width,height,codec_name", "-of", "json", str(path),
    ], capture_output=True, text=True, timeout=45, check=True)
    stream = json.loads(result.stdout)["streams"][0]
    numerator, denominator = map(int, stream["avg_frame_rate"].split("/"))
    assert numerator / denominator == 50, (str(path), stream)
    frames = stream.get("nb_frames")
    assert frames not in (None, "N/A"), f"Container frame count unavailable: {path}"
    assert int(frames) == length, (str(path), frames, length)
    return dict(frames=int(frames), codec=stream["codec_name"],
                width=stream["width"], height=stream["height"])


def audit_pair(dataset, sidecar, cache_root, probe, hash_cache, *, provenance=False, cache_rgb=False):
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from rhsci.data.ardy_sidecar import (
        ArdySidecarReader, dataset_content_fingerprint, validate_processed_dataset_context,
    )

    info = json.loads((dataset / "meta/info.json").read_text())
    episodes = read_lines(dataset / "meta/episodes.jsonl")
    tasks = {item["task_index"] for item in read_lines(dataset / "meta/tasks.jsonl")}
    assert info["fps"] == 50
    assert info["features"]["action"]["shape"] == [36]
    assert info["features"]["states"]["shape"] == [32]
    assert len({e["episode_index"] for e in episodes}) == len(episodes) == info["total_episodes"]
    assert sum(e["length"] for e in episodes) == info["total_frames"]
    context = json.loads((sidecar / "manifest.json").read_text())["context"]
    validate_processed_dataset_context(dataset, context)
    reader = ArdySidecarReader(sidecar, context, require_formal_teacher=True)
    assert set(reader._episodes) == {e["episode_index"] for e in episodes}
    values = {"action": [], "states": []}
    videos, numeric_hashes, duplicates = [], {}, []
    valid_frames, raw_orders, frame_offset = 0, set(), 0
    for ep in episodes:
        eid, length = ep["episode_index"], ep["length"]
        template = dict(episode_index=eid, episode_chunk=eid // info.get("chunks_size", 1000),
                        video_key="observation.images.egocentric")
        path = dataset / info["data_path"].format(**template)
        available = pq.read_schema(path).names
        columns = ["action", "states", "episode_index", "frame_index", "task_index"]
        columns += [k for k in ("timestamp", "index") if k in available]
        table = pq.read_table(path, columns=columns).to_pydict()
        assert len(table["action"]) == length
        assert set(table["episode_index"]) == {eid}
        assert np.array_equal(table["frame_index"], np.arange(length))
        assert set(table["task_index"]) <= tasks
        if "timestamp" in table:
            np.testing.assert_allclose(table["timestamp"], np.arange(length) / 50, rtol=0, atol=2e-5)
        if "index" in table:
            np.testing.assert_array_equal(table["index"], np.arange(frame_offset, frame_offset + length))
        frame_offset += length
        arrays = []
        for key, width in (("action", 36), ("states", 32)):
            array = np.asarray(table[key], np.float32)
            assert array.shape == (length, width) and np.isfinite(array).all(), (eid, key)
            values[key].append(array)
            arrays.append(array)
        numeric_hash = hashlib.sha256(b"".join(a.tobytes() for a in arrays)).hexdigest()
        if numeric_hash in numeric_hashes:
            duplicates.append([numeric_hashes[numeric_hash], eid])
        numeric_hashes[numeric_hash] = eid
        targets = reader._episodes[eid]
        assert targets["ardy_body_latent"].shape == (length, 128)
        assert not targets["ardy_loss_mask"][:6].any(), (eid, "history leakage")
        valid_frames += int(targets["ardy_loss_mask"].sum())
        raw_orders.update(np.unique(np.diff(targets["raw_frame_index"])).tolist())
        video = dataset / info["video_path"].format(**template)
        assert video.is_file() and video.stat().st_size > 0, str(video)
        videos.append((video, length))
    result = dict(dataset=str(dataset), sidecar=str(sidecar), episodes=len(episodes),
                  frames=info["total_frames"], fps=50, splits=info.get("splits"),
                  content_fingerprint=dataset_content_fingerprint(dataset),
                  sidecar_manifest_sha256=sha256(sidecar / "manifest.json"),
                  all_sidecar_hashes_shapes_masks_verified=True, valid_ardy_frames=valid_frames,
                  raw_frame_index_deltas=sorted(raw_orders), duplicate_numeric_episodes=duplicates,
                  teacher={k: context.get(k) for k in ("teacher_verification_status", "teacher_pipeline",
                           "ardy_model_revision", "ardy_source_commit", "formal_reference")},
                  eligible_action_windows={str(h):sum(max(0,e["length"]-h+1) for e in episodes)
                                           for h in (16,30)}, statistics={})
    for key in values:
        array = np.concatenate(values[key]).astype(np.float64)
        computed = dict(min=array.min(0), max=array.max(0), mean=array.mean(0), std=array.std(0),
                        q01=np.quantile(array, .01, axis=0), q99=np.quantile(array, .99, axis=0))
        result["statistics"][key] = {}
        for filename in ("stats.json", "stats_psi0.json"):
            saved = json.loads((dataset / "meta" / filename).read_text())[key]
            metrics = {}
            for metric, expected in computed.items():
                actual = np.asarray(saved[metric], np.float64)
                assert actual.shape == expected.shape and np.isfinite(actual).all()
                metrics[metric] = dict(matches_exact_training_frames=bool(np.allclose(actual, expected, rtol=1e-6, atol=1e-7)),
                                       max_absolute_difference=float(np.max(abs(actual - expected))))
            low, high = np.asarray(saved["min"]), np.asarray(saved["max"])
            assert np.all(high >= low)
            qlo, qhi = np.asarray(saved["q01"]), np.asarray(saved["q99"])
            assert np.all(qhi >= qlo)
            ill = abs(qhi-qlo) < 1e-4 * (abs(qhi)+abs(qlo)+1e-8)
            varies = computed["max"] - computed["min"] > 1e-7
            rare = np.flatnonzero(ill & varies)
            result["statistics"][key][filename] = dict(metrics=metrics,
                out_of_saved_bounds_frames=int(((array < low-1e-6) | (array > high+1e-6)).any(1).sum()),
                quantile_zero_span_but_varies_dimensions=rare.tolist(),
                rare_dimension_nonconstant_frames={str(i):int((abs(array[:,i]-qlo[i])>1e-6).sum()) for i in rare})
    if probe:
        with ThreadPoolExecutor(max_workers=4) as pool:
            streams = list(pool.map(probe_video, videos))
        result["video_container_checks"] = dict(episodes=len(streams), frames=sum(v["frames"] for v in streams),
            formats=sorted({(v["codec"],v["width"],v["height"]) for v in streams}),
            scope="all containers, metadata frame counts; not all decoded pixels")
    if cache_root and (cache_root / dataset.name / "manifest.json").exists():
        from rhsci.data.raw_input_cache import read_manifest

        directory, cache, digest = read_manifest(cache_root, dataset)
        assert cache["fingerprint"] == result["content_fingerprint"]
        columns = torch.load(directory/cache["table"], weights_only=True, mmap=True, map_location="cpu")
        for key in values:
            np.testing.assert_array_equal(columns[key].numpy(), np.concatenate(values[key]))
        payloads = [(directory/n,r) for n,r in cache["files"].items()]
        sources = [(dataset/r["source"],dict(sha256=r["source_sha256"])) for r in cache["episodes"].values()]
        for r in cache["episodes"].values():
            pts = np.load(directory/r["pts"], mmap_mode="r", allow_pickle=False)
            np.testing.assert_allclose(pts, np.arange(r["frames"])/50, rtol=0, atol=1e-6)
        if hash_cache:
            def check_hash(item):
                p, record = item
                assert sha256(p) == record["sha256"], str(p)
            with ThreadPoolExecutor(max_workers=2) as pool:
                list(pool.map(check_hash,payloads+sources))
        result["cache"] = dict(path=str(directory), manifest_sha256=digest,
            numeric_columns_exact=True, all_pts_verified=True, source_stat_verified=True,
            full_payload_and_source_sha_verified=hash_cache, payload_files=len(payloads),
            payload_bytes=sum(r["size"] for _,r in payloads))
    else:
        result["cache"] = dict(selected_shared_cache_present=False)
    if provenance:
        result["provenance"] = audit_provenance(dataset)
    if cache_rgb and result["cache"].get("manifest_sha256"):
        result["cache"]["rgb_decode"] = compare_cache_rgb(dataset, cache_root)
    return result


def integrity_errors(result):
    """数值统计或包装载荷不匹配属于失败；演示独立性/成功性另在报告中解释。"""
    errors = []
    for field, files in result["statistics"].items():
        for filename, value in files.items():
            for metric, record in value["metrics"].items():
                if not record["matches_exact_training_frames"]:
                    errors.append(f"{filename}/{field}/{metric}: differs from actual training frames")
    for record in result.get("provenance", []):
        if record.get("artifacts_mismatched"):
            errors.append(f"{record['file']}: artifact hashes differ: {record['artifacts_mismatched']}")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", nargs=2, type=Path, action="append", required=True,
                        metavar=("DATASET", "SIDECAR"))
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--probe-videos", action="store_true")
    parser.add_argument("--hash-cache", action="store_true")
    parser.add_argument("--verify-provenance", action="store_true")
    parser.add_argument("--compare-cache-rgb", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    report = dict(date_utc=datetime.now(timezone.utc).isoformat(),
                  scope="structural/numeric consistency, not success or held-out certification; no GPU; read-only inputs",
                  datasets=[], errors=[])
    for dataset, sidecar in args.pair:
        print(f"AUDIT {dataset.name}",flush=True)
        try:
            result = audit_pair(dataset.resolve(),sidecar.resolve(),
                args.cache_root,args.probe_videos,args.hash_cache,
                provenance=args.verify_provenance, cache_rgb=args.compare_cache_rgb)
            report["datasets"].append(result)
            failures = integrity_errors(result)
            report["errors"].extend(dict(dataset=str(dataset), error=error) for error in failures)
            print(f"INTEGRITY_{'FAILED' if failures else 'CHECKED'} {dataset.name}; review scientific findings in report",flush=True)
        except Exception as error:
            report["errors"].append(dict(dataset=str(dataset),error=f"{type(error).__name__}: {error}"))
            print(f"FAIL {dataset.name}: {type(error).__name__}: {error}",flush=True)
        args.report.write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n")
    return bool(report["errors"])


if __name__ == "__main__":
    sys.exit(main())
