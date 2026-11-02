from __future__ import annotations

import argparse
import hashlib
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

PACKAGE=Path(__file__).resolve().parents[1]
SCRIPT=PACKAGE/'scripts/nbs/calibrate_nbs_backbone_v093.py'
spec=importlib.util.spec_from_file_location('calibration_test_module',SCRIPT)
calibration=importlib.util.module_from_spec(spec);spec.loader.exec_module(calibration)
fixture_spec=importlib.util.spec_from_file_location('calibration_fixture',Path(__file__).with_name('test_unified_dataset_v093.py'))
fixture_module=importlib.util.module_from_spec(fixture_spec);fixture_spec.loader.exec_module(fixture_module)
dataset=calibration.dataset


def fixture(root):
    build_args,_,_=fixture_module.fixture(root)
    weak=dataset.read_json(root/'data/weak/manifest.json')
    identifiers=['p3','p1','p5','p6']
    digest=hashlib.sha256(('\n'.join(identifiers)+'\n').encode()).hexdigest()
    weak['roles'].append({'role':'core','rows':4,'protein_ids_sha256':digest,'backbone_dense_file':'core.npy'})
    fixture_module.js(root/'data/weak/manifest.json',weak)
    probability=np.array([[.91,.81,.15,.19,.72,.45],[.3,.78,.09,.31,.82,.23],
                          [.2,.1,.87,.25,.6,.17],[.93,.2,.1,.88,.81,.2]],np.float32)
    np.save(root/'data/weak/core.npy',probability)
    args=argparse.Namespace(source_work_dir=Path('work'),output=Path('thresholds.json'),
                           label_holdout_fraction=0.,seed=1337,chunk_rows=2,low_frequency_max=999)
    return args,build_args,probability


class HistogramTests(unittest.TestCase):
    def test_histogram_matches_strict_bruteforce_with_ties_and_endpoints(self):
        p=np.array([[0.,.1,.5,1.],[.5,.5001,.9,.1]],np.float32)
        y=np.array([[0,1,1,0],[0,1,1,0]],bool)
        grid=calibration.threshold_grid()
        h,hp=calibration.histogram_counts(p,y,grid)
        curve=calibration.curve_from_histograms(h,hp)
        expected_pred=np.array([(p.astype(np.float64)>t).sum() for t in grid])
        expected_tp=np.array([((p.astype(np.float64)>t)&y).sum() for t in grid])
        np.testing.assert_array_equal(curve['predicted'],expected_pred)
        np.testing.assert_array_equal(curve['tp'],expected_tp)
        self.assertEqual(curve['predicted'][500],3)
        self.assertEqual(curve['predicted'][-1],0)

    def test_histogram_streaming_matches_single_pass(self):
        rng=np.random.default_rng(99)
        p=rng.integers(0,11,(11,7)).astype(np.float16)/10
        y=rng.random(p.shape)>.8
        first=calibration.histogram_counts(p[:5],y[:5])
        second=calibration.histogram_counts(p[5:],y[5:])
        whole=calibration.histogram_counts(p,y)
        for a,b,c in zip(first,second,whole):np.testing.assert_array_equal(a+b,c)


class CalibrationTests(unittest.TestCase):
    def test_actual_core_role_rows_holdout_and_bruteforce_threshold(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);args,_,p=fixture(root)
            before={str(x):dataset.sha256(x) for x in (root/'data').rglob('*') if x.is_file()}
            result=calibration.calibrate(args,root)
            config=dataset.read_json(root/'work/resolved_config.json')
            _,roles,local,ids=dataset.load_registry(root/'data/features/protein_registry.csv')
            ptr=np.load(root/'data/indices/gold_ptr.npy');go=np.load(root/'data/indices/gold_go.npy')
            train,val=dataset.training_core_split(ids['core'],ptr,config['full_task'])
            counts=dataset.count_gold(ptr,go,train,6)
            mask=counts>0
            mask[5]=mask[0]  # canonical alias may have raw zero frequency
            groups=dataset.frequency_groups(counts)
            gold=np.zeros((len(train),6),bool)
            for row,protein in enumerate(train):gold[row,go[ptr[protein]:ptr[protein+1]]]=True
            for i,(group,_,_) in enumerate(dataset.GROUPS):
                m=(groups==i)&mask
                if i in (1,2,3,4) and gold[:,m].sum():
                    values=p[local[train]][:,m].astype(np.float64);target=gold[:,m]
                    f=[]
                    for threshold in calibration.threshold_grid():
                        predicted=values>threshold
                        f.append(2*(predicted&target).sum()/(predicted.sum()+target.sum()))
                    self.assertEqual(result['thresholds'][group],float(calibration.threshold_grid()[np.argmax(f)]))
            self.assertEqual(result['provenance']['num_excluded_validation_core_proteins'],len(val))
            self.assertFalse(result['independently_calibrated'])
            self.assertFalse(result['provenance']['independent_test_labels_used'])
            self.assertEqual(before,{str(x):dataset.sha256(x) for x in (root/'data').rglob('*') if x.is_file()})
            old=(root/'thresholds.json').read_bytes()
            calibration.calibrate(args,root)
            self.assertEqual(old,(root/'thresholds.json').read_bytes())

    def test_label_holdout_changes_fitting_scope_and_builder_checks_split(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);args,build_args,_=fixture(root)
            args.label_holdout_fraction=.5
            result=calibration.calibrate(args,root)
            self.assertGreater(result['provenance']['num_excluded_task_columns'],1)
            self.assertFalse(result['provenance']['heldout_label_targets_used_for_threshold_fitting'])
            build_args.thresholds_json=root/'thresholds.json'
            with self.assertRaisesRegex(ValueError,'supervision split differs'):
                dataset.build_dataset(build_args,root)
            build_args.label_holdout_fraction=.5
            built=dataset.build_dataset(build_args,root)
            self.assertTrue(built['complete'])

    def test_modelout_groups_stay_fixed_and_nonboundary_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);args,_,_=fixture(root)
            args.low_frequency_max=0
            result=calibration.calibrate(args,root)
            self.assertTrue(all(value==.5 for value in result['thresholds'].values()))
            args.low_frequency_max=17
            with self.assertRaisesRegex(ValueError,'group boundary'):
                calibration.calibrate(args,root)

    def test_installed_pure_cpu_cli(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);fixture(root)
            folder=root/'nbs_models/nbs_protein_go/nbs_pg'
            shutil.copyfile(dataset.MODULE if hasattr(dataset,'MODULE') else PACKAGE/'nbs_models/nbs_protein_go/nbs_pg/unified_weak_dataset_v093.py',
                            folder/'unified_weak_dataset_v093.py')
            cli=root/'scripts/nbs/calibrate_nbs_backbone_v093.py';cli.parent.mkdir(parents=True)
            shutil.copyfile(SCRIPT,cli)
            done=subprocess.run([sys.executable,str(cli),'--source-work-dir',str(root/'work'),
                                  '--output',str(root/'thresholds.json'),'--chunk-rows','2'],capture_output=True,text=True)
            self.assertEqual(done.returncode,0,done.stderr)
            self.assertEqual(dataset.read_json(root/'thresholds.json')['schema'],calibration.SCHEMA)


if __name__=='__main__':unittest.main()
