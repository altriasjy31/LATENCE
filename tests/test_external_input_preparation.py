"""CPU tests for full-cohort preservation and feature/sequence identity."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "nbs"


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.test = self.root / "test.json"
        self.test.write_text(json.dumps({"schema": "latence-unseen-go-test-v1", "complete": True,
            "namespace": "bp", "candidate_go_ids": ["GO:0000001", "GO:0000002"],
            "cohorts": {"example": {"protein_ids": ["B", "A", "C"],
                "evaluable_protein_ids": ["B", "A"], "positives": [
                    {"protein_id": "A", "go_id": "GO:0000001"}]}}}))
        self.fa = self.root / "proteins.fa"
        self.fa.write_text(">A\nACD\n>B\nMKK\n>C\nPPG\n")

    def tearDown(self):
        self.tmp.cleanup()

    def run_prepare(self, *extra):
        return subprocess.run([sys.executable, str(SCRIPTS / "prepare_external_unseen_inputs.py"),
            "--test-manifest", str(self.test), "--cohort", "example", "--fasta", str(self.fa),
            "--output-dir", str(self.root / "output"), *map(str, extra)],
            capture_output=True, text=True)

    def test_full_axis_no_label_leakage(self):
        result = self.run_prepare()
        self.assertEqual(result.returncode, 0, result.stderr)
        table = pd.read_pickle(self.root / "output/metadata.pkl")
        self.assertEqual(list(table.proteins), ["B", "A", "C"])
        self.assertEqual(set(table), {"proteins", "sequences"})
        self.assertEqual((self.root / "output/go_ids.txt").read_text(), "GO:0000001\nGO:0000002\n")

    def test_feature_rows_align_on_sequence_not_source_order(self):
        path = self.root / "source.pkl"
        pd.DataFrame({"proteins": ["C", "A", "B"], "sequences": ["PPG", "ACD", "MKK"],
                      "interpros": [[], ["IPR000001"], ["IPR000002"]],
                      "prop_annotations": [["GO:0000002"]] * 3}).to_pickle(path)
        result = self.run_prepare("--interpro-metadata", path, "--feature-provenance", "snapshot 2022")
        self.assertEqual(result.returncode, 0, result.stderr)
        table = pd.read_pickle(self.root / "output/metadata.pkl")
        self.assertEqual(table.interpros.tolist(), [["IPR000002"], ["IPR000001"], []])
        self.assertNotIn("prop_annotations", table)

    def test_changed_sequence_rejected(self):
        path = self.root / "source.pkl"
        pd.DataFrame({"proteins": ["A", "B", "C"], "sequences": ["ACG", "MKK", "PPG"],
                      "interpros": [[], [], []]}).to_pickle(path)
        result = self.run_prepare("--interpro-metadata", path, "--feature-provenance", "snapshot")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("sequence mismatch", result.stderr)
        self.assertFalse((self.root / "output").exists())

    def test_missing_features_explicit_and_no_protein_drop(self):
        path = self.root / "source.pkl"
        pd.DataFrame({"proteins": ["A"], "sequences": ["ACD"], "interpros": [[]]}).to_pickle(path)
        result = self.run_prepare("--interpro-metadata", path, "--feature-provenance", "snapshot")
        self.assertNotEqual(result.returncode, 0)
        result = self.run_prepare("--interpro-metadata", path, "--feature-provenance", "snapshot",
                                  "--allow-missing-interpro-rows")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(pd.read_pickle(self.root / "output/metadata.pkl")), 3)
        self.assertEqual((self.root / "output/missing_interpro_protein_ids.txt").read_text(), "B\nC\n")

    def test_missing_sequence_fails(self):
        self.fa.write_text(">A\nACD\n>B\nMKK\n")
        result = self.run_prepare()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cohort sequences missing", result.stderr)


if __name__ == "__main__":
    unittest.main()
