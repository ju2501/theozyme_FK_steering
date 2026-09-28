# Lab and Colab runbook

## Lab / remote GPU

If SSH/VPN access already works, the Chromebook can submit work without being
physically in the lab. Scheduler, hostname and shared image paths must come from
the lab; this repository does not invent an sbatch partition or resource limit.

Run the read-only probe in the intended runtime:

```bash
python scripts/probe_runtime.py --out runtime_probe_lab.json \
  --upstream /actual/path/RFdiffusion2 --sif /actual/path/image.sif
```

The paths above are placeholders. For container package versions, run the probe
inside that SIF too, binding this repository at the same absolute path. Keep the
host probe and container probe in separate output files.

Construct the official baseline command using the SHA returned by the probe:

```bash
python pipelines/rfd2/baseline.py \
  --upstream /actual/path/RFdiffusion2 --sif /actual/path/image.sif \
  --commit FULL_UPSTREAM_SHA --out /actual/path/new_baseline_run
```

This defaults to dry-run. Add `--execute` to launch. The wrapper checks a clean
checkout and exact SHA, includes both steer source and upstream in PYTHONPATH,
and writes an image hash, command and log on actual launch. It intentionally
rejects an existing output directory. The wrapper itself has not been GPU-run.
A successful process exit is not evidence that every designed structure passed.

Do not run `scripts/run_steered.sh` expecting RFdiffusion2 results yet. Follow the
integration gate first.

## Colab

Start with `notebooks/colab_cpu.ipynb` to verify this package and the CPU demo.
The notebook includes an optional runtime probe cell. It does not install or
execute RFdiffusion2. Running the notebook on Colab has not been verified here.

For a real GPU run:

1. Select a GPU runtime and record `nvidia-smi`, Python and driver versions.
2. Obtain a pinned upstream checkout and weights using its official instructions.
3. Validate the available environment BEFORE committing to the experiment batch.
   Do not assume the lab SIF will run unchanged in a hosted Colab VM.
4. If container execution is unavailable, use the upstream native environment
   route matched to the actual GPU/runtime. Keep its version pins in a separate
   Colab profile. Do not mix it into the lightweight CPU environment.
5. Run one unchanged upstream baseline. Keep setup logs and returned artifacts.
6. Only then implement/test steering in that same runtime.

The [official installation guide](https://rosettacommons.github.io/RFdiffusion2/installation.html)
provides Apptainer and native environment routes but does not guarantee every
driver/platform combination. Colab resources vary; a paid plan is not a guarantee
of one GPU type or uninterrupted runtime. [Colab FAQ](https://research.google.com/colaboratory/faq.html?hl=en).

Store completed run artifacts outside an ephemeral VM before the session ends.
Use local VM storage while actively writing, then copy each completed batch and
its config/logs. The current demo supports repeat-level restart; true RFdiffusion2
trajectory restart is not implemented.
