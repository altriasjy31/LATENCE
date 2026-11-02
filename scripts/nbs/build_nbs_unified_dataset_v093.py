#!/usr/bin/env python3
"""Build v093 weak labels and frequency-protected candidate evidence on CPU."""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'nbs_models/nbs_protein_go/nbs_pg/unified_weak_dataset_v093.py'
spec = importlib.util.spec_from_file_location('_nbs_unified_dataset_cli', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if __name__ == '__main__':
    module.main()
