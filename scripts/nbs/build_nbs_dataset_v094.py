#!/usr/bin/env python3
"""Build v094 backbone hard labels and modelout soft supervision on CPU."""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / 'nbs_models/nbs_protein_go/nbs_pg/weak_dataset_v094.py'
spec = importlib.util.spec_from_file_location('_nbs_unified_dataset_cli', MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

if __name__ == '__main__':
    module.main()
