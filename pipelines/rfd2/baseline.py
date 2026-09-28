#!/usr/bin/env python3
"""Construct the upstream documented baseline command. GPU execution unverified.

Dry-run by default. --execute explicitly runs it; never installs dependencies.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from steer.io.config import file_hash, atomic_json


def build_command(upstream, sif, out, benchmark, source):
    paths = [Path(x).resolve() for x in (upstream, sif, out, source)]
    upstream, sif, out, source = paths
    if any(',' in str(p) or ':' in str(p) or '\n' in str(p) for p in paths):
        raise ValueError('Apptainer paths cannot contain comma, colon, or newline')
    if not (upstream/'rf_diffusion/benchmark/pipeline.py').is_file():
        raise ValueError('Missing upstream benchmark/pipeline.py')
    if not sif.is_file():
        raise ValueError('SIF does not exist')
    return ['apptainer', 'exec', '--nv',
            '--bind', f'{upstream}:{upstream}:ro', '--bind', f'{source}:{source}:ro',
            '--bind', f'{out}:/runs', '--pwd', str(upstream),
            '--env', f'PYTHONPATH={source}:{upstream}', str(sif),
            'python', str(upstream/'rf_diffusion/benchmark/pipeline.py'),
            '--config-name=open_source_demo', f'sweep.benchmarks={benchmark}', 'outdir=/runs']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', required=True)
    parser.add_argument('--sif', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--commit', required=True, help='Full expected upstream commit SHA')
    parser.add_argument('--benchmark', default='active_site_unindexed_atomic_partial_ligand')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    root, out = Path(args.upstream).resolve(), Path(args.out).resolve()
    source = Path(__file__).resolve().parents[2]/'src'
    try:
        actual = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != args.commit:
            raise ValueError(f'Upstream commit mismatch: actual {actual}')
        status = subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain'], text=True)
        if status.strip():
            raise ValueError('Upstream checkout is dirty; review and record changes before running')
        command = build_command(root, args.sif, out, args.benchmark, source)
        print(shlex.join(command), flush=True)
        if not args.execute:
            return 0
        out.mkdir(parents=True, exist_ok=False)
        manifest = {'status': 'started', 'kind': 'upstream_baseline_gpu_not_prevalidated',
                    'upstream_commit': actual, 'sif_sha256': file_hash(args.sif),
                    'argv': command, 'steering': 'none',
                    'downstream': 'upstream default design stage; no AF2 wrapper'}
        atomic_json(out/'launcher_manifest.json', manifest)
        try:
            with (out/'launcher.log').open('w') as log:
                run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            manifest['returncode'] = run.returncode
            manifest['status'] = 'process_finished' if run.returncode == 0 else 'process_failed'
            returncode = run.returncode
        except OSError as exc:
            manifest['status'], manifest['error'] = 'launch_failed', str(exc)
            returncode = 2
        atomic_json(out/'launcher_manifest.json', manifest)
        return returncode
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
