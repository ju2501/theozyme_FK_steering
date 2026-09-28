# Build verification — 2026-09-24

- Editable package build/install succeeded using the preinstalled NumPy, without downloading dependencies.
- 52 unittest cases passed. Tests are also pytest-compatible; pytest itself was not installed in the build runtime.
- Synthetic PDB score CLI succeeded.
- Full 20-ensemble synthetic benchmark and matplotlib report generation succeeded.
- Resume from completed and partially missing ensemble checkpoints passed. Changed identities and unintended overwrite were rejected.
- Known binary-state terminal tilt matched its analytic target within the statistical test tolerance.
- CPU import does not pull in Torch or matplotlib.
- Python compile and shell syntax checks passed.
- Runtime probe executed successfully in the build environment.
- Baseline launcher command construction tested without executing a container.
- Plot visually inspected: text and panels are readable.

## Not verified

Actual Crostini installation; fresh network dependency resolution; conda creation; Colab notebook execution; GitHub Actions; any GPU/container/model runtime; RFdiffusion2 steering; LigandMPNN or AF2 execution.

## Not implemented

Real RFdiffusion2 adapter, real enzyme task, full downstream pipeline, gradient guidance, trajectory-level checkpoint/restart.

Exact Python/NumPy/source identifiers are recorded in results/manifest.json. No upstream source, weights, real protein data or SIF is included.


## 2026-09-27: v0.1.1 packaging repair

The v0.1.0 ZIP omitted configs/runs/toy.json due to an overbroad runs-directory exclusion. The old archive integrity check did not catch this.

Fixed the archive builder and anchored /runs/ in .gitignore. Added two regression tests. Extracted the rebuilt ZIP into a separate directory and installed THAT source in a new venv using existing system dependencies (no fresh network dependency resolution). Confirmed the import path points to the extracted copy; all 54 tests and the complete synthetic demo with plot passed. git check-ignore confirmed configs/runs/toy.json is trackable. This is still not a test on the user's actual Crostini system.
