"""Original JAR + corrected normalization tests (real Java when available)."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/nbs'))
import normalize_deepgozero_ontology as norm

TINY = '''format-version: 1.2
ontology: go
data-version: test-2025-06-01

[Term]
id: GO:0000001
name: root
namespace: biological_process

[Term]
id: GO:0000002
name: child
namespace: biological_process
is_a: GO:0000001

[Term]
id: GO:0000003
name: defined child
namespace: biological_process
intersection_of: GO:0000001
intersection_of: part_of GO:0000002

[Typedef]
id: part_of
name: part of
is_transitive: true
'''


class NormalizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo = Path(os.environ.get('DEEPGOZERO_SOURCE', str(ROOT.parents[1] / 'deepgozero_original_work/source/deepgozero')))
        if not (cls.repo / 'Normalizer.groovy').is_file():
            raise unittest.SkipTest('Set DEEPGOZERO_SOURCE to the attached original repository')

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.obo = self.path / 'tiny.obo'
        self.obo.write_text(TINY)

    def tearDown(self):
        self.temp.cleanup()

    def test_dependency_block_removed_without_algorithm_change(self):
        original = (self.repo / 'Normalizer.groovy').read_text()
        changed = norm.dependency_free_source(original)
        self.assertNotIn('@Grab', changed)
        self.assertIn('new IntegerOntologyObjectFactoryImpl()', changed)
        self.assertEqual(changed, original[original.index('])') + 2:].lstrip())

    def test_local_jars_include_obo_and_jcel(self):
        paths, records, providers = norm.jar_inventory(self.repo / 'jars')
        self.assertEqual(len(providers['OBO_parser']), 1)
        self.assertEqual(len(providers['JCEL']), 1)
        self.assertEqual(len(paths), len(records))
        self.assertTrue(all(len(row['sha256']) == 64 for row in records))

    def test_obo_imports_rejected_not_xrefs(self):
        self.obo.write_text(TINY + '\nxref: http://example.org/reference\n')
        self.assertEqual(norm.input_inventory(self.obo)['obo_intersection_of_lines'], 2)
        self.obo.write_text('format-version: 1.2\nimport: http://example.org/live.owl\n')
        with self.assertRaisesRegex(ValueError, 'imports'):
            norm.input_inventory(self.obo)

    def test_parser_rejects_silently_unrecognized_subclass(self):
        p = self.path / 'bad.norm'
        p.write_text('GO_0000001 SubClassOf GO_0000002\nLONG_UNKNOWN_CLASS SubClassOf GO_0000002\n')
        with self.assertRaisesRegex(ValueError, 'ignored'):
            norm.parse_normal_forms(self.repo, p)

    def test_preflight_without_runtime_creates_no_output(self):
        result = subprocess.run([sys.executable, str(ROOT / 'scripts/nbs/normalize_deepgozero_ontology.py'),
            '--repo-dir', str(self.repo), '--ontology-file', str(self.obo), '--output-dir', str(self.path / 'out'),
            '--java-bin', '/nonexistent/java', '--dry-run'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['mode'], 'corrected')
        self.assertIsNone(report['java_executable'])
        self.assertFalse((self.path / 'out').exists())

    @unittest.skipUnless(shutil.which('java'), 'Java runtime unavailable')
    def test_actual_java_obo_to_four_normal_forms_and_coverage(self):
        candidates = self.path / 'go.txt'
        candidates.write_text('GO:0000003\nGO:9999999\n')
        out = self.path / 'out'
        command = [sys.executable, str(ROOT / 'scripts/nbs/normalize_deepgozero_ontology.py'),
                   '--repo-dir', str(self.repo), '--ontology-file', str(self.obo),
                   '--candidate-go-ids', str(candidates), '--output-dir', str(out)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads((out / 'normalization_manifest.json').read_text())
        self.assertTrue(report['complete'])
        self.assertEqual(report['mode'], 'corrected')
        self.assertEqual(report['status'], 'normalized')
        self.assertEqual(report['normal_form_counts'], {'nf1': 2, 'nf2': 1, 'nf3': 1, 'nf4': 1})
        self.assertEqual(report['candidate_go_supported'], 1)
        self.assertEqual(report['candidate_go_missing'], ['GO:9999999'])
        self.assertEqual(report['backend_counts']['excluded_nonmodel_axioms'], 1)
        lines = (out / 'go.norm').read_text().splitlines()
        self.assertIn('GO_0000001 and ZC_0000000 SubClassOf GO_0000003', lines)
        self.assertIn('part_of some GO_0000002 SubClassOf ZC_0000000', lines)
        self.assertNotIn('GO_0000001 and GO_0000002 SubClassOf GO_0000003', lines)
        self.assertFalse((out / 'native_diagnostic.norm').exists())
        rerun = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(rerun.returncode, 0)
        self.assertIn('Choose a new normalization directory', rerun.stderr)


if __name__ == '__main__':
    unittest.main()
