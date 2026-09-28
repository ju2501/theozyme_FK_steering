import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('baseline_launcher', ROOT/'pipelines/rfd2/baseline.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    def test_dry_command_preserves_paths_and_upstream_import(self):
        with tempfile.TemporaryDirectory(prefix='path with spaces ') as directory:
            root = Path(directory)
            upstream, source = root/'upstream', root/'source'
            (upstream/'rf_diffusion/benchmark').mkdir(parents=True)
            (upstream/'rf_diffusion/benchmark/pipeline.py').write_text('# fixture')
            source.mkdir()
            sif = root/'image.sif'
            sif.write_bytes(b'fake image, never execute')
            out = root/'new run'
            command = launcher.build_command(upstream, sif, out, 'demo', source)
            self.assertIn(f'PYTHONPATH={source}:{upstream}', command)
            self.assertIn(f'{out}:/runs', command)
            self.assertEqual(command[-1], 'outdir=/runs')
            self.assertFalse(out.exists())

    def test_missing_upstream_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Missing upstream'):
            launcher.build_command('/does-not-exist', '/absent.sif', '/new-out', 'demo', ROOT/'src')
