"""Synthetic v088 graph audit: detect seen-weak self-node reentry."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.dark_weak.audit_v088_graph_inputs import audit


class FakeData:
    def __init__(self,config,stores=None):pass
    def ensure_prepared(self):pass
    def set_sampling_context(self,step,rank,training):pass
    def inference_batch(self,input_dir,rows,device='cpu'):
        row=int(rows[0])
        orig=-row-1
        # The fake graph contains an original training weak ID 42 as support.
        ids=torch.tensor([orig,42],dtype=torch.long)
        block={
            'src_global_ids':ids,
            'dst_global_ids':torch.tensor([orig],dtype=torch.long),
            'edge_index':torch.tensor([[1],[0]],dtype=torch.long),
            'edge_type':torch.tensor([0],dtype=torch.long),
        }
        return {'sampled_global_ids':ids,
                'sampled_seed_index':torch.tensor([0],dtype=torch.long),
                'sampled_gold_edge':torch.empty((2,0),dtype=torch.long),
                'sampled_pseudo_edge':torch.empty((2,0),dtype=torch.long),
                'blocks':[block]}


class AuditTests(unittest.TestCase):
    def test_direct_seen_weak_reentry_flagged(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory);inp=p/'input';inp.mkdir()
            (inp/'protein_ids.txt').write_text('P\n')
            (inp/'ind_test_input_manifest.json').write_text('{}')
            (p/'config.json').write_text('{}')
            np.save(p/'originals.npy',np.array([42],dtype=np.int64))
            args=argparse.Namespace(config=p/'config.json',input_dir=inp,
                target_original_global_ids=p/'originals.npy',batch_size=1,limit=1,
                all_proteins=True,device='cpu',out=p/'audit.json')
            fake_data=types.ModuleType('nbs_pg.full_task_data_v088')
            fake_data.FullTaskDataV088=FakeData
            fake_loader=types.ModuleType('nbs_pg.local_loader_v088')
            fake_loader.build_latence_nbs_stores=lambda c: None
            fake_v084=types.ModuleType('nbs_pg.full_task_data_v084')
            fake_v084.RELATIONS=('cosine','ppi')
            # Original dependencies are not imported by this narrow audit.
            with patch.dict(sys.modules,{'nbs_pg.full_task_data_v088':fake_data,
                'nbs_pg.local_loader_v088':fake_loader,
                'nbs_pg.full_task_data_v084':fake_v084}):
                result=audit(args)
            self.assertEqual(result['self_support_hits'],1)
            self.assertEqual(result['self_support_status'],'detected_REQUIRES_GRAPH_INPUT_FIX')
            self.assertEqual(result['ghost_edges_noncosine'],0)

if __name__=='__main__':unittest.main()
