#!/usr/bin/env python3
"""Independent P0 runner; preserve the original runner and strict checkpoints.

Validate sampler equivalence on the target data before training. Original
checkpoints cannot be passed directly to this runner's RESUME: fast checkpoints
record both this entrypoint and the sampler implementation. Use a new work dir
for a fresh fast run; same-phase resume requires a fast checkpoint.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "nbs_models/nbs_protein_go")]

# Original runners remain at scripts/nbs, even if this additive entrypoint is
# deployed under nbs/scripts. This import deliberately preserves their globals.
from scripts.nbs import train_nbs_full_task_v089 as base  # noqa: E402
from nbs_pg import full_task_sampler_fast as fast  # noqa: E402

_ORIGINAL_ATTRIBUTE = "_nbs_p0_original_prediction_implementation"
_WRAPPER_ATTRIBUTE = "_nbs_p0_fast_wrapper_path"
_base_prediction_implementation = getattr(
    base, _ORIGINAL_ATTRIBUTE, base.prediction_implementation)


def prediction_implementation(variant):
    # Copy the original dictionary: never mutate a cached baseline identity.
    result = dict(_base_prediction_implementation(variant))
    result["full_task_sampler_fast.py"] = fast.implementation_sha256()
    result[Path(__file__).name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


def install():
    """Install once or reload the same wrapper without wrapping itself."""
    wrapper = str(Path(__file__).resolve())
    installed = getattr(base, _WRAPPER_ATTRIBUTE, None)
    if installed is not None and installed != wrapper:
        raise RuntimeError("a different P0 entrypoint is already installed in this process")
    original = getattr(base, _ORIGINAL_ATTRIBUTE, None)
    if original is None:
        if base.prediction_implementation is not _base_prediction_implementation:
            raise RuntimeError("base runner prediction implementation changed before P0 installation")
        setattr(base, _ORIGINAL_ATTRIBUTE, _base_prediction_implementation)
    elif original is not _base_prediction_implementation:
        raise RuntimeError("P0 original implementation identity changed")
    base.FullTaskData = fast.FullTaskDataV088Fast
    base.prediction_implementation = prediction_implementation
    setattr(base, _WRAPPER_ATTRIBUTE, wrapper)


if __name__ == "__main__":
    install()
    base.main()
