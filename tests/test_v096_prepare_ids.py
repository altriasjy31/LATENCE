from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.nbs.prepare_esm_ids_v096 import parser, prepare, registry_ids


class PrepareIdsTests(unittest.TestCase):
    def test_union_alignment_and_missing_fasta(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            reg = root / 'registry.csv'
            reg.write_text('protein_idx,protein_id,role,role_row_idx\n0,A,core,0\n1,B,weak,0\n')
            inp = root / 'test'; inp.mkdir()
            (inp / 'protein_ids.txt').write_text('B\nC\n')
            cache = root / 'raw'; cache.mkdir()
            (cache / 'A.npy').write_bytes(b'placeholder')
            fasta = root / 'all.fasta'; fasta.write_text('>B\nMAC\n>C\nMDD\n')
            out = root / 'ids'
            args = parser().parse_args(['--registry', str(reg), '--input-dir', str(inp),
                '--features-dir', str(cache), '--fasta', str(fasta), '--output-dir', str(out)])
            report = prepare(args)
            self.assertEqual(report['num_proteins'], 3)
            self.assertEqual((out / 'protein_ids.txt').read_text(), 'A\nB\nC\n')
            self.assertEqual((out / 'missing_embeddings.fasta').read_text(), '>B\nMAC\n>C\nMDD\n')
            with self.assertRaises(FileExistsError):
                prepare(args)

    def test_duplicate_registry_or_incomplete_fasta_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); reg = root / 'reg.csv'
            reg.write_text('protein_idx,protein_id,role,role_row_idx\n0,A,core,0\n1,A,weak,0\n')
            with self.assertRaises(ValueError):
                registry_ids(reg)
            reg.write_text('protein_idx,protein_id,role,role_row_idx\n0,A,core,0\n')
            fasta = root / 'all.fasta'; fasta.write_text('>B\nMAC\n')
            args = parser().parse_args(['--registry', str(reg), '--fasta', str(fasta), '--output-dir', str(root / 'out')])
            with self.assertRaises(ValueError):
                prepare(args)
            self.assertFalse((root / 'out').exists())


if __name__ == '__main__':
    unittest.main()
