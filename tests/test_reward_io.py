import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from steer.io.config import load_json, load_task
from steer.io.pdb import read_pdb, select_atoms
from steer.geometry.theozyme import Theozyme, Constraint
from steer.rewards.theozyme_reward import TheozymeReward

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "configs/theozyme/synthetic.json"
PDB = ROOT / "tests/fixtures/synthetic.pdb"


class RewardIOTests(unittest.TestCase):
    def setUp(self):
        self.task = load_task(TASK)
        self.coords = select_atoms(read_pdb(PDB), self.task.atoms)
        self.reward = TheozymeReward(self.task)

    def test_fixture_is_ideal(self):
        score = self.reward(self.coords)
        self.assertTrue(score.valid)
        self.assertTrue(score.passed)
        self.assertEqual(score.value, 0)

    def test_perturbation_degrades_reward(self):
        self.coords['d'] += [5, 0, 0]
        score = self.reward(self.coords)
        self.assertFalse(score.passed)
        self.assertLess(score.value, -1)

    def test_missing_atom_is_invalid(self):
        del self.coords['d']
        score = self.reward(self.coords)
        self.assertFalse(score.valid)
        self.assertIsNone(score.to_dict()['reward'])

    def test_nan_is_invalid(self):
        self.coords['d'][0] = np.nan
        self.assertFalse(self.reward(self.coords).valid)

    def test_fixed_only_task_cannot_steer(self):
        data = load_json(TASK)
        for atom in data['atoms'].values():
            atom['role'] = 'fixed'
        with self.assertRaisesRegex(ValueError, 'fixed'):
            Theozyme.from_dict(data).validate_for_steering()

    def test_fixed_terms_are_diagnostics(self):
        data = load_json(TASK)
        data['constraints'][-1]['target'] = 50
        score = TheozymeReward(Theozyme.from_dict(data))(self.coords)
        self.assertEqual(score.value, 0)
        self.assertFalse(score.passed)

    def test_bad_scale(self):
        with self.assertRaises(ValueError):
            Constraint('bad', 'distance', ('a','b'), 1, 0, 0)

    def test_unknown_atom(self):
        data = load_json(TASK)
        data['constraints'][0]['atoms'] = ['a', 'unknown']
        with self.assertRaises(ValueError):
            Theozyme.from_dict(data)

    def test_typo_rejected(self):
        data = load_json(TASK)
        data['typo'] = True
        with self.assertRaises(ValueError):
            Theozyme.from_dict(data)

    def _read_text(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.pdb'
            path.write_text(text)
            return read_pdb(path)

    def test_pdb_only_first_model(self):
        body = PDB.read_text().replace('END\n', '')
        atoms = self._read_text('MODEL        1\n' + body + 'ENDMDL\nMODEL        2\n' + body + 'ENDMDL\n')
        self.assertEqual(len(atoms), 4)

    def test_altloc_highest_occupancy(self):
        line = next(x for x in PDB.read_text().splitlines() if x.startswith('HETATM'))
        a = line[:16] + 'A' + line[17:54] + '  0.30' + line[60:]
        b = line[:16] + 'B' + line[17:30] + '   9.000' + line[38:54] + '  0.70' + line[60:]
        atoms = self._read_text(a + '\n' + b + '\n')
        self.assertEqual(next(iter(atoms.values()))[0], 9)

    def test_duplicate_conformer_rejected(self):
        line = next(x for x in PDB.read_text().splitlines() if x.startswith('HETATM'))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self._read_text(line + '\n' + line)

    def test_insertion_code_preserved(self):
        line = next(x for x in PDB.read_text().splitlines() if x.startswith('HETATM'))
        atoms = self._read_text(line[:26] + 'B' + line[27:])
        self.assertEqual(next(iter(atoms)).icode, 'B')

    def test_ambiguous_selector_rejected(self):
        with self.assertRaises(ValueError):
            select_atoms(read_pdb(PDB), {'all': {'chain': 'X'}})

    def test_missing_selector_rejected(self):
        with self.assertRaises(ValueError):
            select_atoms(read_pdb(PDB), {'x': {'name': 'NO'}})
