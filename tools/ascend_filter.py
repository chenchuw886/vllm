#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Generate a conservative Ascend-available test list.

Heuristics (default):
- Keep pytest.mark.cpu_test.
- Exclude GPU-heavy directories.
- Exclude files mentioning CUDA/ROCm/XPU keywords or GPU-specific patterns.

This does NOT guarantee Ascend compatibility; it is a safe starting point.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path


EXCLUDE_DIRS_DEFAULT = {
    "tests/cuda",
    "tests/rocm",
    "tests/kernels",
    "tests/compile",
    "tests/distributed",
    "tests/kv_offload",
}

KEYWORD_RE = re.compile(
    r"\b("
    r"cuda|cudagraph|cublas|cudnn|cutlass|flash_attn|flashmla|triton|"
    r"nvfp4|nvml|trtllm|rocm|hip|xpu|ipex"
    r")\b",
    re.IGNORECASE,
)

GPU_PATTERNS = [
    re.compile(r"torch\.cuda\."),
    re.compile(r'device\s*=\s*["\']cuda'),
    re.compile(r"current_platform\.is_cuda\(\)"),
    re.compile(r"current_platform\.is_rocm\(\)"),
    re.compile(r"current_platform\.is_xpu\(\)"),
    re.compile(r"pytest\.mark\.skipif\([^\n]*current_platform\.is_cuda\(\)"),
    re.compile(r"pytest\.mark\.skipif\([^\n]*current_platform\.is_rocm\(\)"),
    re.compile(r"pytest\.mark\.skipif\([^\n]*current_platform\.is_xpu\(\)"),
    re.compile(r"VLLM_TARGET_DEVICE"),
]

CPU_MARK = re.compile(r"pytest\.mark\.cpu_test")


def iter_test_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*.py"):
        name = path.name
        if name.startswith("test_") or name.endswith("_test.py"):
            files.append(path)
    return files


def should_exclude_dir(path: Path, exclude_dirs: set[str]) -> bool:
    for d in exclude_dirs:
        dpath = Path(d)
        try:
            path.relative_to(dpath)
            return True
        except ValueError:
            continue
    return False


def is_cpu_marked(content: str) -> bool:
    return CPU_MARK.search(content) is not None


def is_gpu_like(content: str) -> bool:
    if KEYWORD_RE.search(content):
        return True
    return any(p.search(content) for p in GPU_PATTERNS)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Ascend test list.")
    parser.add_argument(
        "--root",
        default="tests",
        help="Test root directory (default: tests)",
    )
    parser.add_argument(
        "--output",
        default="tests/ascend_available.txt",
        help="Output file path (default: tests/ascend_available.txt)",
    )
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Additional directories to exclude (repeatable).",
    )
    parser.add_argument(
        "--no-default-excludes",
        action="store_true",
        help="Do not use the built-in GPU-heavy directory excludes.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    output = Path(args.output)
    exclude_dirs = set() if args.no_default_excludes else set(EXCLUDE_DIRS_DEFAULT)
    exclude_dirs.update(args.exclude_dir)

    candidates: set[str] = set()
    for path in iter_test_files(root):
        rel = path.as_posix()
        if should_exclude_dir(path, exclude_dirs):
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        if is_cpu_marked(content):
            candidates.add(rel)
            continue

        if is_gpu_like(content):
            continue

        candidates.add(rel)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(sorted(candidates)) + "\n", encoding="utf-8")
    print(f"Wrote {len(candidates)} entries to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
