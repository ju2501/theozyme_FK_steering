#!/usr/bin/env python3
"""Package source and small results while retaining configs/runs/."""
import argparse
from pathlib import Path
import zipfile

ROOT_EXCLUSIONS = {'runs', '.venv', '.git', 'build', 'dist', 'data'}
ANY_EXCLUSIONS = {'__pycache__', '.pytest_cache', '.ipynb_checkpoints'}
REQUIRED = ('pyproject.toml', 'src/steer/__init__.py',
            'configs/runs/toy.json', 'configs/theozyme/synthetic.json')


def build_release(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    for name in REQUIRED:
        if not (root / name).is_file():
            raise ValueError(f'Missing required release file: {name}')
    files = []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root)
        if relative.parts[0] in ROOT_EXCLUSIONS:
            continue
        if any(p in ANY_EXCLUSIONS or p.endswith('.egg-info') for p in relative.parts):
            continue
        if relative.parts[:2] == ('configs', 'local'):
            continue
        if path.name == '.env' or path.name.startswith(('.env.', 'runtime_probe')):
            continue
        if path.suffix in ('.pyc', '.pyo', '.sif', '.pt', '.pth', '.ckpt', '.safetensors', '.zip'):
            continue
        if path.is_symlink():
            raise ValueError(f'Release must not contain symlinks: {relative}')
        if path.is_file() and path != output:
            files.append(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path('theozyme-steer') / path.relative_to(root))
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive integrity check failed')
        for name in REQUIRED:
            if f'theozyme-steer/{name}' not in archive.namelist():
                raise ValueError(f'Required file missing from ZIP: {name}')
    return len(files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    count = build_release(Path(__file__).resolve().parents[1], args.out)
    print(f'Packaged {count} files into {args.out}')
