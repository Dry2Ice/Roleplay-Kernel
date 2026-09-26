"""Raise the project version everywhere it is declared.

Usage: python scripts/bump_version.py 0.2.0

The previous version is discovered rather than hardcoded, so the script keeps
working after the first bump. Each target is rewritten by parsing its own
syntax instead of a blind regex, because a mangled manifest or pyproject file
breaks the build in ways that are tedious to diagnose later.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

VERSION_RE = re.compile(r"\d+\.\d+\.\d+")

PYPROJECT_VERSION_RE = re.compile(r'(?m)^(version\s*=\s*)"[^"]*"(\s*)$')
SIDECAR_VERSION_RE = re.compile(r'(?m)^(SIDECAR_VERSION\s*=\s*)"[^"]*"(\s*)$')
EXT_VERSION_RE = re.compile(r"(?m)^(const EXTENSION_VERSION\s*=\s*)'[^']*'(;.*)$")


def bump(path: pathlib.Path, pattern: re.Pattern[str], version: str, label: str) -> None:
    if not path.exists():
        print(f"skipped (missing): {path}")
        return
    text = path.read_text(encoding="utf-8")
    updated, count = pattern.subn(lambda m: f"{m.group(1)}{label}{m.group(2)}", text, count=1)
    if count == 0:
        print(f"skipped (no version line): {path}")
        return
    path.write_text(updated, encoding="utf-8")
    print(f"updated: {path}")


def main() -> None:
    version = sys.argv[1] if len(sys.argv) > 1 else "0.1.1"
    if not VERSION_RE.fullmatch(version):
        raise SystemExit(f"invalid version: {version}")

    bump(
        pathlib.Path("pyproject.toml"),
        PYPROJECT_VERSION_RE,
        version,
        f'"{version}"',
    )
    bump(
        pathlib.Path("src/roleplay_kernel/sidecar.py"),
        SIDECAR_VERSION_RE,
        version,
        f'"{version}"',
    )
    bump(
        pathlib.Path("index.js"),
        EXT_VERSION_RE,
        version,
        f"'{version}'",
    )

    manifest = pathlib.Path("manifest.json")
    if manifest.exists():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["version"] = version
        manifest.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"updated: {manifest}")
    print("bumped to", version)


if __name__ == "__main__":
    main()
