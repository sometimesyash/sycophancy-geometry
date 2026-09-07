"""Enforce the rule that keeps this project runnable after Azure access ended.

Scripts numbered 10+ and the modules they depend on must be pure functions of files on
disk. If a network dependency creeps back in, whoever inherits this repository cannot run
it, and the research stops. This check is cheap; run it before every commit.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_MODULES = {"azclient", "httpx", "requests", "urllib.request", "aiohttp", "openai", "azure"}
PORTABLE_SRC = {"geometry.py", "factual.py", "corpus.py", "multiturn.py", "judge.py"}


def imports_of(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        print(f"  ! {path.name}: syntax error {exc}")
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def main() -> int:
    failures: list[str] = []

    portable_scripts = sorted(
        p for p in (ROOT / "scripts").glob("*.py")
        if p.stem[:2].isdigit() and int(p.stem[:2]) >= 10
    )
    print(f"Checking {len(portable_scripts)} portable scripts")
    for p in portable_scripts:
        bad = {m for m in imports_of(p) if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_MODULES)}
        if bad:
            failures.append(f"{p.relative_to(ROOT)} imports {sorted(bad)}")
        print(f"  {'FAIL' if bad else 'ok  '}  {p.name}")

    print("\nChecking portable src modules")
    for name in sorted(PORTABLE_SRC):
        p = ROOT / "src" / name
        if not p.exists():
            continue
        bad = {m for m in imports_of(p) if any(m == f or m.startswith(f + ".") for f in FORBIDDEN_MODULES)}
        if bad:
            failures.append(f"src/{name} imports {sorted(bad)}")
        print(f"  {'FAIL' if bad else 'ok  '}  {name}")

    print("\nChecking frozen data is present")
    required = [
        "corpus_labelled.parquet",
        "corpus_mt_labelled.parquet",
        "corpus_fc_labelled.parquet",
    ]
    for f in required:
        p = ROOT / "data" / "frozen" / f
        ok = p.exists()
        if not ok:
            failures.append(f"missing frozen data: {f}")
        print(f"  {'ok  ' if ok else 'FAIL'}  {f}")

    if failures:
        print("\nPORTABILITY CHECK FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nPORTABILITY CHECK PASSED: downstream work runs without Azure.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
