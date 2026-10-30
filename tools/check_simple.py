"""只读核验外部 SIMPLE 官方版本、源码及递归子模块；不下载或启动仿真。"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

LOCK = Path(__file__).resolve().parents[1] / "configs/simple/official.json"


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def verify_clean(root):
    # CFS 上原生 .so 符号链接可被 status 误报为 M；diff 重新比较真实内容。
    # 分别检查暂存区/工作区；子模块逐个检查，不能用忽略子模块掩盖生产代码变化。
    for args in (("diff", "HEAD"), ("diff", "--cached", "HEAD")):
        subprocess.run(
            ["git", "-C", str(root), *args, "--quiet", "--ignore-submodules=all"],
            check=True,
        )
    if git(root, "ls-files", "--others", "--exclude-standard"):
        raise ValueError("SIMPLE 或其子模块存在额外文件：" + str(root))


def check(root):
    root = Path(root).resolve(strict=True)
    lock = json.loads(LOCK.read_text())
    errors = []
    # 必须核对 Git 根目录，避免误把父仓库或其他安装当成 SIMPLE。
    if Path(git(root, "rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("路径必须指向独立 SIMPLE checkout 根目录")
    revision = git(root, "rev-parse", "HEAD")
    if revision != lock["revision"]:
        errors.append(f"SIMPLE HEAD: {revision}; expected {lock['revision']}")
    verify_clean(root)
    for name, record in lock["files"].items():
        path = root / name
        digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        # LFS 图片允许官方指针或其同 SHA 的实体；两者均不代表运行资产已齐备。
        valid_lfs = (
            "lfs_sha256" in record
            and digest == record["lfs_sha256"]
            and path.stat().st_size == record["lfs_bytes"]
        )
        if digest != record["sha256"] and not valid_lfs:
            errors.append("SIMPLE 文件不匹配：" + name)
    for name, expected in lock["submodules"].items():
        # 未初始化目录的 rev-parse 会向父目录查找，因此同时核对其 Git 根目录。
        sub = root / name
        if (
            not sub.exists()
            or Path(git(sub, "rev-parse", "--show-toplevel")).resolve() != sub
            or git(sub, "rev-parse", "HEAD") != expected
        ):
            errors.append("官方子模块缺失或版本不匹配：" + name)
    # 保留 status 输出首字符：'-' 未初始化、'+' 错版本、'U' 冲突，均不能通过。
    statuses = subprocess.check_output(
        ["git", "-C", str(root), "submodule", "status", "--recursive"], text=True
    ).splitlines()
    for line in statuses:
        if not line.startswith(" "):
            errors.append("递归子模块未就绪：" + line)
        else:
            # 官方路径无空格；第三段可含描述性 tag，不能将其当作路径。
            verify_clean(root / line.split()[1])
    if errors:
        raise ValueError("\n".join(errors))
    return dict(
        source=str(root),
        revision=revision,
        files_verified=len(lock["files"]),
        recursive_submodules_verified=len(statuses),
        scope="source only; simulator environment, LFS assets and GPU rollouts not validated",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="独立的官方 SIMPLE checkout 根目录")
    args = parser.parse_args()
    try:
        result = check(args.source)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
