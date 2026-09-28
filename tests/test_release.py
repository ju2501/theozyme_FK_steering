import importlib.util
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_builder', ROOT/'scripts/build_release.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ReleaseTests(unittest.TestCase):
    def test_run_configuration_survives_packaging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)/'project'
            for name in (*builder.REQUIRED, 'runs/toy/large-result.json'):
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture')
            output = Path(directory)/'release.zip'
            builder.build_release(root, output)
            with zipfile.ZipFile(output) as archive:
                self.assertIn('theozyme-steer/configs/runs/toy.json', archive.namelist())
                self.assertNotIn('theozyme-steer/runs/toy/large-result.json', archive.namelist())

    def test_gitignore_anchors_output_directory(self):
        lines = (ROOT/'.gitignore').read_text().splitlines()
        self.assertIn('/runs/', lines)
        self.assertNotIn('runs/', lines)
