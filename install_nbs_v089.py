#!/usr/bin/env python3
"""Install v0891, upgrading only approved files with known predecessor hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

VERSION_METADATA = frozenset((
    "nbs_models/nbs_protein_go/nbs_pg/__init__.py",
    "nbs_models/nbs_protein_go/pyproject.toml",
))
RELEASE_UPGRADES = frozenset((
    "scripts/nbs/train_nbs_full_task_v089.py",
    "scripts/nbs/eval_nbs_full_task_v089.py",
    "scripts/nbs/check_nbs_v089_acceptance.py",
    "scripts/nbs/run_nbs_v089.sh",
    "nbs_models/nbs_protein_go/nbs_pg/full_task_phase_v089.py",
    "nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.9_mb6400_single_ep100.json",
    "nbs_models/nbs_protein_go/configs/bp_full_task_v0.8.9_mb12800_ddp_ep100.json",
    "install_nbs_v089.py", "README_NBS_v089.md",
    "RELEASE_NBS_v089.json", "VALIDATION_NBS_v089.json",
))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(bundle, target):
    manifest = json.loads((bundle / "MANIFEST_NBS_v089.json").read_text())
    if manifest.get("release_version") != "0.8.9.1":
        raise ValueError("Unexpected bundle version")
    copies, unchanged, upgrades = [], [], []
    for entry in manifest["files"]:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe bundle path: " + str(relative))
        source, destination = bundle / relative, target / relative
        if digest(source) != entry["sha256"]:
            raise ValueError("Bundle bytes differ from manifest: " + str(relative))
        if entry.get("install") is False:
            continue
        try:
            destination.resolve().relative_to(target.resolve())
        except ValueError as exc:
            raise ValueError("Destination resolves outside the project: " + str(relative)) from exc
        if destination.exists():
            current = digest(destination) if destination.is_file() else None
            predecessors = entry.get("upgrade_from_sha256", [])
            if isinstance(predecessors, str):
                predecessors = [predecessors]
            if (current != entry["sha256"] and relative.as_posix() in VERSION_METADATA | RELEASE_UPGRADES
                    and current in predecessors and current is not None):
                copies.append((source, destination))
                upgrades.append(relative)
                continue
            if current != entry["sha256"]:
                raise FileExistsError("Existing file differs; nothing was installed: " + str(destination))
            unchanged.append(relative)
        else:
            copies.append((source, destination))
    return copies, unchanged, upgrades


def install(bundle, target, dry_run=False):
    bundle, target = Path(bundle).resolve(), Path(target).resolve()
    copies, unchanged, upgrades = preflight(bundle, target)
    if not dry_run:
        for source, destination in copies:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + ".v089-install.tmp")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
    return {"release_version": "0.8.9.1", "dry_run": dry_run,
            "new_files": len(copies) - len(upgrades), "identical_existing_files": len(unchanged),
            "version_metadata_upgrades": [p.as_posix() for p in upgrades if p.as_posix() in VERSION_METADATA],
            "release_file_upgrades": [p.as_posix() for p in upgrades if p.as_posix() in RELEASE_UPGRADES],
            "target": str(target)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(install(Path(__file__).resolve().parent, args.project_root, args.dry_run), indent=2))


if __name__ == "__main__":
    main()
