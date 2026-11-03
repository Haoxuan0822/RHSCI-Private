"""Compare restored scene/material files and aliases with verified official ZIP inventories."""

import json
import zlib
from pathlib import Path
from zipfile import ZipFile

ROOT = Path("/root/workspace/Haoxuan/RHSCI-runtime")
DATA = ROOT / "external/SIMPLE-official/data"


def fingerprint(path):
    checksum = 0
    with path.open("rb") as stream:
        while block := stream.read(4 * 1024 * 1024):
            checksum = zlib.crc32(block, checksum)
    return path.stat().st_size, checksum


def main():
    expected = {}
    failures = []
    archives = sorted((ROOT / "simple-assets-archives").glob("scenes_hssd_*.zip"))
    archives.append(ROOT / "simple-assets-archives/vMaterials_2.zip")
    for archive in archives:
        with ZipFile(archive) as zip_file:
            for member in zip_file.infolist():
                if member.is_dir():
                    continue
                relative = member.filename
                path = DATA / relative
                signature = (member.file_size, member.CRC)
                if path in expected and expected[path] != signature:
                    failures.append(dict(path=str(path), reason="conflicting archive members"))
                expected[path] = signature
    for path, signature in expected.items():
        if not path.is_file() or fingerprint(path) != signature:
            failures.append(dict(path=str(path), reason="missing or different from official ZIP"))
    actual = {
        p
        for folder in ("scenes/hssd", "vMaterials_2")
        for p in (DATA / folder).rglob("*")
        if p.is_file()
    }
    missing = sorted(map(str, expected.keys() - actual))
    extra = sorted(map(str, actual - expected.keys()))
    aliases = json.loads((ROOT / "logs/pi05-between-usd-aliases.json").read_text())
    alias_results = []
    for item in aliases:
        alias = Path(item["alias"])
        source = Path(item["usd"]).parent / alias.name
        source_files = {p.relative_to(source): p for p in source.rglob("*") if p.is_file()}
        alias_files = {p.relative_to(alias): p for p in alias.rglob("*") if p.is_file()}
        differences = sorted(map(str, source_files.keys() ^ alias_files.keys()))
        if alias.resolve() != source.resolve():
            differences.extend(
                str(key)
                for key in source_files.keys() & alias_files.keys()
                if fingerprint(source_files[key]) != fingerprint(alias_files[key])
            )
        alias_results.append(
            dict(
                alias=str(alias),
                source=str(source),
                files=len(source_files),
                differences=differences,
            )
        )
    report = dict(
        archives=len(archives),
        expected_files=len(expected),
        actual_files=len(actual),
        missing=missing,
        extra=extra,
        content_failures=failures,
        aliases=alias_results,
        method="Exact filename sets, byte sizes and ZIP CRC32; official ZIP SHA256 was verified during restoration. No files or material bindings modified.",
    )
    (ROOT / "logs/material-inventory-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    summary = {
        k: v
        for k, v in report.items()
        if k not in ("aliases", "missing", "extra", "content_failures")
    }
    summary.update(
        missing_count=len(missing), extra_count=len(extra), content_failure_count=len(failures)
    )
    summary["alias_count"] = len(alias_results)
    summary["alias_mismatches"] = sum(bool(item["differences"]) for item in alias_results)
    print(json.dumps(summary, indent=2), flush=True)
    raise SystemExit(bool(missing or extra or failures or summary["alias_mismatches"]))


if __name__ == "__main__":
    main()
