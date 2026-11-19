"""Exhaustive toy check of the corrected NF1-NF4 class normalization.

Run with DEEPGOZERO_SOURCE=/path/to/deepgozero python -m unittest discover
  -s tests -p 'test_corrected_normalizer_semantics.py'
The Java integration uses only the supplied JARs and a temporary OBO file.
It verifies this finite toy, not general equivalence of arbitrary GO-Plus OWL.
"""
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TOY = """format-version: 1.2
ontology: go
data-version: semantic-test

[Term]
id: GO:0000001
name: root process
namespace: biological_process

[Term]
id: GO:0000002
name: child process
namespace: biological_process
is_a: GO:0000001

[Term]
id: GO:0000003
name: intersection process
namespace: biological_process
intersection_of: GO:0000001
intersection_of: part_of GO:0000002

[Typedef]
id: part_of
name: part of
is_transitive: true
"""


def some(role, values):
    return sum(1 << i for i in range(2)
               if any(role & (1 << (i * 2 + j)) and values & (1 << j)
                      for j in range(2)))


def parse(path):
    axioms, classes, roles = [], set(), set()
    for line in Path(path).read_text().splitlines():
        line = line.strip().replace('_', ':')
        if ' SubClassOf ' not in line:
            continue
        left, right = line.split(' SubClassOf ')
        if ' and ' in left:
            a, b = left.split(' and ')
            axioms.append(('nf2', a, b, right))
            classes.update([a, b, right])
        elif ' some ' in left:
            role, a = left.split(' some ')
            axioms.append(('nf3', role, a, right))
            classes.update([a, right]); roles.add(role)
        elif ' some ' in right:
            role, b = right.split(' some ')
            axioms.append(('nf4', left, role, b))
            classes.update([left, b]); roles.add(role)
        else:
            axioms.append(('nf1', left, right)); classes.update([left, right])
    return axioms, classes, roles


def satisfies(axioms, values, relations):
    for axiom in axioms:
        if axiom[0] == 'nf1':
            left, right = values[axiom[1]], values[axiom[2]]
        elif axiom[0] == 'nf2':
            left, right = values[axiom[1]] & values[axiom[2]], values[axiom[3]]
        elif axiom[0] == 'nf3':
            left, right = some(relations[axiom[1]], values[axiom[2]]), values[axiom[3]]
        else:
            left, right = values[axiom[1]], some(relations[axiom[2]], values[axiom[3]])
        if left & ~right:
            return False
    return True


class CorrectedNormalizerSemantics(unittest.TestCase):
    def test_original_factory_collision_has_a_real_counterexample(self):
        # Domain {a,b}, C1={a,b}, C2={b}, C3={a}, r={(a,b)}.
        c1, c2, c3, role = 3, 2, 1, 2
        self.assertEqual(c3, c1 & some(role, c2))
        self.assertEqual(c2 & ~c1, 0)
        # Broken normalizer conflates a fresh existential class with C2.
        self.assertNotEqual(some(role, c2) & ~c2, 0)
        self.assertNotEqual((c1 & c2) & ~c3, 0)

    def test_corrected_java_preserves_all_two_object_class_interpretations(self):
        repo = Path(os.environ.get('DEEPGOZERO_SOURCE',
            str(ROOT.parents[1] / 'deepgozero_original_work/source/deepgozero')))
        if not (repo / 'jars').is_dir() or shutil.which('java') is None:
            self.skipTest('Set DEEPGOZERO_SOURCE and install Java 11+ to run supplied-JAR integration')
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            ontology = temp / 'tiny.obo'
            ontology.write_text(TOY)
            output = temp / 'normalized'
            run = subprocess.run([sys.executable,
                str(ROOT / 'scripts/nbs/normalize_deepgozero_ontology.py'),
                '--repo-dir', str(repo), '--ontology-file', str(ontology),
                '--output-dir', str(output)], capture_output=True, text=True, timeout=60)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            manifest = json.loads((output / 'normalization_manifest.json').read_text())
            self.assertTrue(manifest['complete'])
            self.assertEqual(manifest['mode'], 'corrected')
            self.assertGreater(manifest['backend_counts']['excluded_nonmodel_axioms'], 0)
            axioms, classes, roles = parse(output / 'go.norm')
            named = ['GO:0000001', 'GO:0000002', 'GO:0000003']
            self.assertTrue(set(named).issubset(classes))
            self.assertEqual(len(roles), 1)
            role_name = next(iter(roles))
            auxiliary = sorted(classes - set(named))
            self.assertEqual(len(auxiliary), 1)
            checked = 0
            for values in itertools.product(range(4), repeat=3):
                for role in range(16):
                    c1, c2, c3 = values
                    expected = not (c2 & ~c1) and c3 == (c1 & some(role, c2))
                    actual = any(satisfies(axioms,
                        dict(zip(named + auxiliary, values + extra)), {role_name: role})
                        for extra in itertools.product(range(4), repeat=len(auxiliary)))
                    self.assertEqual(actual, expected, (values, role))
                    checked += 1
            self.assertEqual(checked, 1024)


if __name__ == '__main__':
    unittest.main()
