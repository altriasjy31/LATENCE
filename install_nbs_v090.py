#!/usr/bin/env python3
"""Install additive v090 files after complete source/target conflict checks."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install(source, destination, *, dry_run=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("extract the delivery outside the active project")
    manifest = json.loads((source / "INSTALL_MANIFEST_v090.json").read_text())
    additions, dependencies = manifest["additions"], manifest["dependencies"]
    planned, identical = [], []
    for group, entries in (("dependency", dependencies), ("v090", additions)):
        for name, expected in entries.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("invalid relative installation path")
            original, target = source / relative, destination / relative
            if not original.is_file() or digest(original) != expected:
                raise ValueError(f"delivery bytes differ: {name}")
            if target.exists():
                if not target.is_file() or digest(target) != expected:
                    raise FileExistsError(f"local {group} differs; preserved without writes: {name}")
                identical.append(name)
            else:
                planned.append(name)
    # All source and destination checks precede any write. Existing different
    # old or new files are never overwritten, including unknown local fixes.
    if not dry_run:
        for name in planned:
            original, target = source / name, destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".v090-install-tmp")
            if temporary.exists():
                raise FileExistsError(f"temporary installation path already exists: {temporary}")
            try:
                shutil.copy2(original, temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    return {"release": "0.9.0", "dry_run": dry_run,
            "new_files": planned, "identical_preserved": identical,
            "existing_files_rewritten": 0}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    print(json.dumps(install(Path(__file__).resolve().parent, args.project_root,
        dry_run=args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
