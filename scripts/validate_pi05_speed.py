"""单张空闲GPU上的完整π0.5短测；不启动守护、不保存正式断点、不操作其他任务。"""

import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path


def admit_gpu(index, shared=False):
    # 在CUDA初始化之前准入；显存之外同时检查计算占用，避免挤占训练/评测。
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i",
            str(index),
            "--query-gpu=memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    used, total, utilization = map(int, result.stdout.strip().split(","))
    if (
        total < 30000
        or (not shared and (used > 1024 or utilization > 5))
        or (shared and total - used < 17500)
    ):
        raise RuntimeError(
            f"GPU {index} busy or too small: used={used}MiB total={total}MiB util={utilization}%; no CUDA test started"
        )
    return (total - used - 1024) / total


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--dataset", type=Path)
    p.add_argument("--sidecar", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--tokenizer", type=Path)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument(
        "--shared", action="store_true", help="授权共享GPU短测；限制分配预算并给其他任务保留1GiB"
    )
    p.add_argument("--steps", type=int, default=4, choices=range(3, 9))
    p.add_argument("--output", type=Path, required=True, help="数据盘内全新JSON报告路径")
    args = p.parse_args()
    if args.batch_size < 1:
        p.error("batch-size must be positive")
    root = Path(os.environ.get("PSI_RUNTIME_ROOT", "/opt/rsci-data"))
    destination = root / "validation/pi05"
    if not args.output.resolve().is_relative_to(destination.resolve()) or args.output.exists():
        p.error(f"output must be a new path under {destination}")
    fraction = admit_gpu(args.gpu, args.shared)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    import torch

    from rsci.backbones.pi05.checkpoints import atomic_json
    from rsci.backbones.pi05.model import Pi05RSCI
    from rsci.backbones.pi05.processing import collate
    from rsci.backbones.pi05.train import parser, preflight, resolve, simple_lr_multiplier

    torch.set_num_threads(1)
    if args.shared:
        torch.cuda.set_per_process_memory_fraction(fraction)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    config = parser().parse_args(["--task", args.task, "--gpus", "0", "--check"])
    for name in ("dataset", "sidecar", "checkpoint", "tokenizer"):
        if getattr(args, name) is not None:
            setattr(config, name, getattr(args, name))
    paths = resolve(config)
    _, dataset = preflight(config, paths)
    # 从第6帧以后开始，保证既含历史，又含正式ARDY；所有profile复用同一CPU输入。
    batches = [
        collate([dataset[6 + step * args.batch_size + i] for i in range(args.batch_size)])
        for step in range(args.steps)
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(
        status="running",
        gpu=args.gpu,
        task=args.task,
        batch=args.batch_size,
        steps=args.steps,
        precision="BF16-autocast/FP32-trainable",
        profiles={},
        scope="single GPU; excludes DDP and loader/serialization throughput; no formal checkpoint",
    )
    atomic_json(args.output, report)
    reference_gradients = None
    reference_final = None
    reference_moments = None
    reference_losses = None
    reference_rng = None

    def compare(actual, expected, *, exact=False, weights=False):
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0 if exact else (1e-5 if weights else 0.03),
            atol=0 if exact else (2e-6 if weights else 5e-4),
        )
        return (actual.float() - expected.float()).abs().max().item()

    try:
        for profile in ("reference", "query", "history", "both"):
            torch.manual_seed(292285)
            torch.cuda.manual_seed_all(292285)
            model = Pi05RSCI(compile_inference=False, execution=profile)
            model.load_initial(paths["checkpoint"])
            model.cuda().train()
            parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}
            optimizer = torch.optim.AdamW(
                parameters.values(),
                lr=1e-4,
                betas=(0.9, 0.95),
                eps=1e-8,
                weight_decay=1e-10,
                fused=True,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, simple_lr_multiplier)
            times, losses, rngs = [], [], []
            max_gradient_error = 0.0
            torch.cuda.reset_peak_memory_stats()
            for step, cpu_batch in enumerate(batches):
                batch = {k: v.cuda() for k, v in cpu_batch.items()}
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                started = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    result = model(**batch)
                result["loss"].backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters.values(), 1.0)
                if not torch.isfinite(result["loss"]) or not torch.isfinite(norm):
                    raise FloatingPointError(f"Nonfinite profile={profile} step={step}")
                optimizer.step()
                scheduler.step()
                torch.cuda.synchronize()
                times.append(time.perf_counter() - started)
                losses.append({k: float(v) for k, v in result.items()})
                rngs.append((torch.get_rng_state(), torch.cuda.get_rng_state()))
                # 梯度在step()后仍保留；第一步比较所有裁剪后梯度，后续比较轨迹终态。
                if step == 0:
                    gradients = {n: p.grad.cpu().clone() for n, p in parameters.items()}
                    if profile == "reference":
                        reference_gradients = gradients
                    else:
                        for name, value in gradients.items():
                            max_gradient_error = max(
                                max_gradient_error,
                                compare(value, reference_gradients[name], exact=profile == "query"),
                            )
                    del gradients
            final = {n: p.detach().cpu().clone() for n, p in parameters.items()}
            moments = {
                n: {
                    k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
                    for k, v in optimizer.state[p].items()
                }
                for n, p in parameters.items()
            }
            max_weight_error = 0.0
            if profile == "reference":
                reference_final, reference_moments = final, moments
                reference_losses, reference_rng = losses, rngs
            else:
                for name, value in final.items():
                    max_weight_error = max(
                        max_weight_error,
                        compare(
                            value, reference_final[name], exact=profile == "query", weights=True
                        ),
                    )
                    for key, state in moments[name].items():
                        compare(state, reference_moments[name][key], exact=profile == "query")
                for expected, actual in zip(reference_losses, losses):
                    for key in expected:
                        compare(
                            torch.tensor(actual[key]),
                            torch.tensor(expected[key]),
                            exact=profile == "query",
                        )
                for expected, actual in zip(reference_rng, rngs):
                    for x, y in zip(expected, actual):
                        compare(x, y, exact=True)
            report["profiles"][profile] = dict(
                update_seconds=times,
                steady_mean_seconds=sum(times[1:]) / (len(times) - 1),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                max_clipped_gradient_abs_error=max_gradient_error,
                max_weight_abs_error=max_weight_error,
                losses=losses,
            )
            atomic_json(args.output, report)
            print(json.dumps({profile: report["profiles"][profile]}), flush=True)
            del model, optimizer, scheduler, parameters, final, moments, batch, result
            gc.collect()
            torch.cuda.empty_cache()
        report["status"] = "passed"
        atomic_json(args.output, report)
    except Exception as exc:
        report.update(status="failed", error=repr(exc))
        atomic_json(args.output, report)
        raise


if __name__ == "__main__":
    main()
