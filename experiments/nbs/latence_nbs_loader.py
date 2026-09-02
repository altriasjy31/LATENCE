from __future__ import annotations

from typing import Any

from nbs_pg.local_loader import build_latence_nbs_train_loader


def build_train_loader(config: dict[str, Any]):
    """Production LATENCE mmap/CSR -> local NBS batch factory."""
    return build_latence_nbs_train_loader(config)
