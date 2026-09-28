import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from steer.cli.benchmark import benchmark
from steer.io.config import load_json, atomic_json

ROOT = Path(__file__).resolve().parents[1]


class CLITests(unittest.TestCase):
    def test_score_cli(self):
        run = subprocess.run([sys.executable, '-m', 'steer.cli.main', 'score', '--task',
                              str(ROOT/'configs/theozyme/synthetic.json'), '--pdb',
                              str(ROOT/'tests/fixtures/synthetic.pdb')], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertTrue(json.loads(run.stdout)['passed'])

    def test_real_adapter_fails_honestly(self):
        run = subprocess.run([sys.executable, '-m', 'steer.cli.main', 'rfd2'], capture_output=True, text=True)
        self.assertEqual(run.returncode, 2)
        self.assertIn('not implemented', run.stderr)

    def test_core_does_not_import_torch(self):
        run = subprocess.run([sys.executable, '-c',
                              "import sys; import steer.particles.fk_steering; import steer.rewards.theozyme_reward; "
                              "assert 'torch' not in sys.modules; assert 'matplotlib' not in sys.modules"], capture_output=True, text=True)
        self.assertEqual(run.returncode, 0, run.stderr)

    def test_benchmark_checkpoint_resume_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            config, out = Path(directory)/'config.json', Path(directory)/'out'
            data = load_json(ROOT/'configs/runs/toy.json')
            data.update(task=str(ROOT/'configs/theozyme/synthetic.json'), replicates=2)
            data['fk'].update(num_particles=8, steps=6, every=2)
            atomic_json(config, data)
            first = benchmark(config, out)
            original_csv = (out/'selected.csv').read_bytes()
            self.assertEqual(first['fk']['model_step_calls_per_replicate'], 48)
            self.assertEqual(first['best_of_n']['model_step_calls_per_replicate'], 48)
            self.assertEqual(benchmark(config, out, resume=True), first)
            self.assertEqual((out/'selected.csv').read_bytes(), original_csv)
            (out/'replicates/0001.json').unlink()
            self.assertEqual(benchmark(config, out, resume=True), first)
            self.assertEqual((out/'selected.csv').read_bytes(), original_csv)
            with self.assertRaises(FileExistsError):
                benchmark(config, out)
            data['seed'] += 1
            atomic_json(config, data)
            with self.assertRaisesRegex(ValueError, 'identity'):
                benchmark(config, out, resume=True)
