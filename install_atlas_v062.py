#!/usr/bin/env python3
"""Install the additive v062 evaluation update into an existing LATENCE project."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil


NEW_FILES = (
    'scripts/dark_weak/eval_atlas_v062.py',
    'scripts/dark_weak/streaming_metric_pack_v062.py',
    'scripts/dark_weak/atlas_precision_v062.py',
    'tests/test_atlas_full_runner_v062.py',
    'tests/test_streaming_metrics_v062.py',
    'tests/test_atlas_precision_v062.py',
    'README_ATLAS_FULL_V062.md',
    'VALIDATION_V062.json',
)
SUPPORT_FILES = (
    'scripts/dark_weak/eval_atlas_v060.py',
    'scripts/dark_weak/atlas_evidence_v060.py',
    'scripts/dark_weak/eval_atlas_v061.py',
    'scripts/dark_weak/align_atlas_expert_v061.py',
    'scripts/dark_weak/complete_metric_pack_v054.py',
    'scripts/dark_weak/eval_stage1_frequency_v053.py',
    'scripts/dark_weak/stage1_evalperf_original_v054.py',
    'tests/test_atlas_runner_v060.py',
)
REQUIRED_PROJECT_FILES = (
    'scripts/eval_dark_weak_predictions.py',
    'scripts/dark_weak/native_v085_metrics.py',
    'experiments/eval_weak_ind_test_detr_diagnostics.py',
    'nbs_models/nbs_protein_go/nbs_pg/full_task_metrics_v085.py',
    'utils/util_functions.py',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def install(project, source=None):
    source = Path(source or __file__).resolve()
    source = source.parent if source.is_file() else source
    project = Path(project).resolve(strict=True)
    if project == source:
        raise ValueError('--project-root must be the existing LATENCE project, not the update directory')
    missing = [rel for rel in REQUIRED_PROJECT_FILES if not (project / rel).is_file()]
    if missing:
        raise FileNotFoundError('This is an update for the existing LATENCE dark-weak project. Missing: ' + ', '.join(missing))
    plan = list(NEW_FILES) + [rel for rel in SUPPORT_FILES if not (project / rel).exists()]
    for rel in plan:
        if not (source / rel).is_file():
            raise FileNotFoundError(f'Update package incomplete: {source / rel}')
        if (project / rel).exists() and not (project / rel).is_file():
            raise ValueError(f'Destination is not a regular file: {project / rel}')
    backup = project / 'backups' / ('atlas_v062_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    report = []
    for rel in plan:
        src, dst = source / rel, project / rel
        expected = digest(src)
        if dst.exists() and digest(dst) == expected:
            report.append(dict(path=rel, action='unchanged', sha256=expected))
            continue
        if dst.exists():
            saved = backup / rel
            saved.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(dst, saved)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if digest(dst) != expected:
            raise RuntimeError(f'Installed bytes differ: {dst}')
        report.append(dict(path=rel, action='installed', sha256=expected))
    return dict(project=str(project), backup=str(backup) if backup.exists() else None,
                legacy_sources_preserved=True, files=report)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, default=Path('.'))
    args = parser.parse_args(argv)
    print(json.dumps(install(args.project_root), indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
