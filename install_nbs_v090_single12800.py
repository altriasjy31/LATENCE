#!/usr/bin/env python3
"""Install the single-GPU v090/12800 patch after checking all local bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def install(source, destination, *, dry_run=False):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("extract the delivery outside the active project")
    if not destination.is_dir():
        raise ValueError("--project-root must be an existing project directory")
    manifest = json.loads((source / "INSTALL_MANIFEST_v090_single12800.json").read_text())
    if manifest.get("revision") != "0.9.0-single12800-r1":
        raise ValueError("unexpected installation manifest")
    entries = {**manifest["dependencies"], **manifest["files"]}
    allowed = manifest.get("allowed_previous_sha256", {})
    operations, preserved = [], []
    for name, expected in entries.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("invalid relative installation path")
        original, target = source / relative, destination / relative
        if not original.is_file() or sha256(original) != expected:
            raise ValueError(f"delivery bytes differ: {name}")
        if target.is_symlink():
            raise FileExistsError(f"local symlink preserved: {name}")
        action = "new"
        if target.exists():
            if not target.is_file():
                raise FileExistsError(f"local non-file preserved: {name}")
            current = sha256(target)
            if current == expected:
                preserved.append(name)
                continue
            if current != allowed.get(name):
                raise FileExistsError(f"local modifications preserved; no files written: {name}")
            action = "upgrade"
        temporary = target.with_name(target.name + ".single12800-install-tmp")
        backup = target.with_name(target.name + ".before-single12800-r1.bak")
        if temporary.exists():
            raise FileExistsError(f"installation temporary already exists: {temporary}")
        if action == "upgrade" and backup.exists() and (not backup.is_file() or sha256(backup) != allowed[name]):
            raise FileExistsError(f"existing backup differs; preserved: {backup}")
        operations.append((name, action, temporary, backup))
    # All conflicts, source checks and backup checks precede any write.
    if not dry_run:
        for name, action, temporary, backup in operations:
            original, target = source / name, destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if action == "upgrade" and not backup.exists():
                shutil.copy2(target, backup)
            try:
                shutil.copy2(original, temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
    return {"revision": manifest["revision"], "dry_run": dry_run,
            "new_files": [name for name, action, _, _ in operations if action == "new"],
            "upgraded_known_v090_files": [name for name, action, _, _ in operations if action == "upgrade"],
            "identical_preserved": preserved}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(install(Path(__file__).resolve().parent, args.project_root,
        dry_run=args.dry_run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
