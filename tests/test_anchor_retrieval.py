"""Run: python -m unittest discover -s tests -p 'test_anchor_retrieval.py'."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/nbs/eval_unseen_go_retrieval_v095.py"
sys.path.insert(0, str(SCRIPT.parent))
from eval_unseen_go_retrieval_v095 import compare_positive_ranks

NEW = ["GO:0000003", "GO:0000004", "GO:0000005"]
OLD = ["GO:0000001", "GO:0000002"]
IDS = ["A", "B", "C", "D"]


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prediction(folder, ids, gos, rows, values, expanded):
    folder.mkdir()
    p = folder / "protein_ids.txt"
    p.write_text("\n".join(ids) + "\n", encoding="utf-8")
    targets = [dict(output_column=i, requested_go_id=g, canonical_go_id=g, ontology_row=r,
                    readout="single_shared_GO_scorer") for i, (g, r) in enumerate(zip(gos, rows))]
    if expanded:
        for target in targets:
            target.update(task_columns=[], seen_task_columns=[], stage2_supervised=False,
                          zero_shot_scope="Stage2_unsupervised_target_only")
    with (folder / "target_go.tsv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(targets[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(targets)
    probability = folder / "probabilities.npy"
    np.save(probability, np.asarray([values[i] for i in ids], dtype=np.float32))
    result = dict(schema="nbs-v095-requested-go-prediction-v1", complete=True,
                  checkpoint_epoch=160, checkpoint_step=7680,
                  num_proteins=len(ids), num_target_GO=len(gos), targets=targets,
                  probability_file=probability.name, output_probability_sha256=sha(probability),
                  protein_ids_file=p.name, target_go_file="target_go.tsv",
                  identity=dict(checkpoint_sha256="a" * 64, resolved_config_sha256="b" * 64,
                                input_manifest_sha256="c" * 64, contract={"task": "bp"},
                                prediction_implementation="predict_nbs_go_v095",
                                protein_ids_sha256=sha(p), requested_go_ids=gos,
                                target_ontology_rows=rows,
                                ontology_bundle_sha256="d" * 64 if expanded else None),
                  ontology_extension=dict(schema="nbs-v095-anchored-ontology-replacement-v1",
                                          old_coordinate_anchors_verified=True,
                                          new_trainable_parameter_rows_added=False,
                                          num_ontology_go=6) if expanded else None)
    save(folder / "prediction_manifest.json", result)
    return result


class AnchorRetrievalTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.extension = self.folder / "extension"
        self.extension.mkdir()
        ext = self.extension / "extension_manifest.json"
        save(ext, dict(schema="latence-v095-anchored-new-go-v1", complete=True,
                       ontology_only=True, task="bp", namespaces=["bp"], old_rows=3, new_rows=3,
                       artifacts={"ontology_bundle.pt": {"sha256": "d" * 64}},
                       sources={"v095_bindings": {"v095_config": {"sha256": "b" * 64}}},
                       fit={"per_term": [dict(go_id=g, row=r, anchor_rows=a) for g, r, a in
                                          zip(NEW, [3, 4, 5], [[1, 2], [2], [1]])]}))
        self.test = self.folder / "test.json"
        self.test_value = dict(schema="latence-unseen-go-test-v1", complete=True, namespace="bp",
                               candidate_go_ids=NEW, candidate_ontology_rows=[3, 4, 5],
                               ontology_bundle_sha256="d" * 64, v095_resolved_config_sha256="b" * 64,
                               sources={"extension_manifest": {"path": str(ext), "sha256": sha(ext)}},
                               cohorts={"dark_weak_all_exp": {
                                   "protein_ids": IDS, "evaluable_protein_ids": IDS[:3],
                                   "input_manifest_sha256": "c" * 64,
                                   "positives": [{"protein_id": "A", "go_id": NEW[0],
                                                  "evidence_codes": ["IDA"]}]}})
        save(self.test, self.test_value)
        self.new, self.old = self.folder / "new", self.folder / "old"
        prediction(self.new, ["D", "C", "B", "A"], NEW, [3, 4, 5],
                   dict(A=[.4, .1, .2], B=[.2, .6, .6], C=[.4, .9, .8], D=[.8, .1, .3]), True)
        self.old_manifest = prediction(self.old, ["C", "A", "D", "B"], OLD[::-1], [2, 1],
                                       dict(A=[.8, .2], B=[.1, .3], C=[.2, .3], D=[.5, .5]), False)

    def tearDown(self):
        self.temp.cleanup()

    def run_cli(self, name, *args, error=None):
        output = self.folder / name
        process = subprocess.run([sys.executable, str(SCRIPT), "--test-manifest", str(self.test),
                                  "--cohort", "dark_weak_all_exp", "--predictions-dir", str(self.new),
                                  "--extension-dir", str(self.extension),
                                  "--anchor-predictions-dir", str(self.old), "--top-k", "1", "2",
                                  "--output-dir", str(output), *args], capture_output=True, text=True)
        if error is not None:
            self.assertNotEqual(process.returncode, 0)
            self.assertIn(error, process.stderr)
            self.assertFalse(output.exists())
            return
        self.assertEqual(process.returncode, 0, process.stderr)
        return output, json.loads((output / "retrieval.json").read_text())

    def test_shuffled_axes_ties_and_pair_delta(self):
        output, result = self.run_cli("full")
        with (output / "positive_pairs_comparison.tsv").open() as f:
            pairs = list(csv.DictReader(f, delimiter="\t"))
        self.assertEqual(len(pairs), 1)
        pair = pairs[0]
        self.assertEqual(pair["protein_id"], "A")
        self.assertEqual(float(pair["expanded_rank_min"]), 2)
        self.assertEqual(float(pair["expanded_rank_max"]), 3)
        self.assertEqual(float(pair["anchor_midrank"]), 1)
        self.assertEqual(float(pair["midrank_improvement_over_anchor"]), -1.5)
        self.assertEqual(float(pair["top@2_credit_improvement_over_anchor"]), -.5)
        self.assertEqual(result["paired_positive_rank_summary"]["n_midrank_worsened"], 1)
        self.assertEqual(result["expanded"]["n_proteins_ranked_per_go"], 4)
        self.assertEqual(result["expanded"]["observed_positive_go_count"], 1)
        self.assertEqual(result["expanded"]["candidate_go_count"], 3)
        _, eligible = self.run_cli("eligible", "--protein-pool", "evaluable")
        self.assertEqual(eligible["expanded"]["n_proteins_ranked_per_go"], 3)
        self.assertEqual(eligible["expanded"]["pair_observed_recall@1"], .5)

    def test_empty_labels(self):
        self.test_value["cohorts"]["dark_weak_all_exp"]["positives"] = []
        save(self.test, self.test_value)
        output, result = self.run_cli("empty")
        self.assertEqual(result["paired_positive_rank_summary"]["n_observed_positive_pairs"], 0)
        self.assertIsNone(result["paired_positive_rank_summary"]["mean_midrank_improvement_over_anchor"])
        self.assertTrue((output / "positive_pairs_comparison.tsv").exists())

    def test_mismatched_checkpoint_rejected(self):
        self.old_manifest["identity"]["checkpoint_sha256"] = "e" * 64
        save(self.old / "prediction_manifest.json", self.old_manifest)
        self.run_cli("bad_checkpoint", error="differ in checkpoint_sha256")

    def test_missing_protein_rejected(self):
        ids = self.old / "protein_ids.txt"
        ids.write_text("C\nA\nD\n")
        self.old_manifest["num_proteins"] = 3
        self.old_manifest["identity"]["protein_ids_sha256"] = sha(ids)
        save(self.old / "prediction_manifest.json", self.old_manifest)
        self.run_cli("missing_protein", error="full cohort")

    def test_corrupted_scores_rejected(self):
        with (self.old / "probabilities.npy").open("ab") as f:
            f.write(b"altered")
        self.run_cli("corrupt", error="SHA256 differs")

    def test_wrong_anchor_ontology_rejected(self):
        self.old_manifest["identity"]["ontology_bundle_sha256"] = "d" * 64
        save(self.old / "prediction_manifest.json", self.old_manifest)
        self.run_cli("bad_ontology", error="original ontology")

    def test_missing_or_duplicate_positive_pair_rejected(self):
        pair = {"protein_id": "A", "go_id": NEW[0]}
        positives = {"A": {NEW[0]: {}}}
        with self.assertRaisesRegex(ValueError, "exactly every observed"):
            compare_positive_ranks([pair], [], positives, {}, {}, 4, [1])
        with self.assertRaisesRegex(ValueError, "Duplicate positive"):
            compare_positive_ranks([pair, pair], [pair], positives, {}, {}, 4, [1])


if __name__ == "__main__":
    unittest.main()
