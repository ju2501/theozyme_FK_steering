#!/usr/bin/env python3
"""Read-only runtime inventory. Does not download models or submit GPU jobs."""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(args, cwd=None):
    if shutil.which(args[0]) is None:
        return {"available": False}
    try:
        run = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=30)
        return {"available": True, "returncode": run.returncode,
                "stdout": run.stdout[:12000], "stderr": run.stderr[:2000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"available": True, "error": str(exc)}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='runtime_probe.json')
    parser.add_argument('--upstream', help='Local RFdiffusion2 checkout')
    parser.add_argument('--sif', help='Optional image to hash (can take time)')
    args = parser.parse_args()
    info = {'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
            'executables': {name: shutil.which(name) for name in ('python', 'git', 'apptainer', 'nvidia-smi')},
            'gpu': command(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader']),
            'apptainer': command(['apptainer', '--version']), 'packages': {}}
    for name in ('numpy', 'torch', 'hydra-core', 'dgl', 'torchdata', 'theozyme-steer'):
        try:
            info['packages'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            info['packages'][name] = None
    if args.upstream:
        root = Path(args.upstream).resolve()
        if not (root/'rf_diffusion').is_dir():
            parser.error('Not an RFdiffusion2 checkout: missing rf_diffusion/')
        info['upstream'] = {'path': str(root), 'commit': command(['git', 'rev-parse', 'HEAD'], root),
                            'status': command(['git', 'status', '--porcelain'], root), 'candidate_hooks': []}
        for path in sorted((root/'rf_diffusion').rglob('*.py')):
            try:
                tree = ast.parse(path.read_text())
            except (OSError, SyntaxError, UnicodeError):
                continue
            functions = [{'name': node.name, 'line': node.lineno} for node in ast.walk(tree)
                         if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                         and any(key in node.name.lower() for key in ('sample', 'denois', 'inferen', 'step'))]
            if functions:
                info['upstream']['candidate_hooks'].append({'file': str(path.relative_to(root)),
                                                            'sha256': sha256(path), 'functions': functions})
    if args.sif:
        path = Path(args.sif).resolve()
        if not path.is_file():
            parser.error('SIF does not exist')
        info['sif'] = {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha256(path)}
    path = Path(args.out)
    # Fail instead of clobbering a previous machine's evidence.
    with path.open('x') as stream:
        json.dump(info, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
