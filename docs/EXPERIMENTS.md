# Experiment accounting

## Executed: synthetic benchmark

20 independent seed runs, N=16 particles, S=40 stochastic coordinate steps,
K=4 selected outputs. This benchmark's only role is to test the implementation.
It contains no protein network, sequence design or fold prediction.

Each seed produces one unguided pool and one FK pool. Unguided returns K uniform
slots; best-of-N takes the best K from the SAME unguided pool. FK returns K uniform
slots after terminal resampling. FK duplicates remain in the dataset, since
discarding them would conceal degeneracy. Selection choices use a separate RNG.

Each strategy uses N*S=640 sampler calls per seed. The actual combined benchmark
executes 1280 calls per seed because the two baseline summaries share their pool.
It is not three independently generated pools. FK evaluates reward at its
scheduled checkpoints plus the terminal step; the baseline only at the end.
Final reporting additionally scores the selected coordinates. This is cheap in
the toy but belongs in real end-to-end cost accounting.

`sampler_elapsed_seconds` includes run_particles work and selection within that
engine. It excludes final report generation. Best-of-N's recorded time shares
baseline sampling and does not include its small external sort. Do not treat
these times as a precise equal-total-compute comparison.

Statistics are based on ensemble-level means. The bootstrap resamples the
paired per-seed FK minus best-of-N differences, not individual particle rows.
The 95% interval is an exploratory percentile bootstrap, not a biological claim.
Current output: mean reward difference +0.9512, interval [0.6129, 1.2831].
Reward improves while coordinate spread falls. Overall tolerance pass counts
are only 0/80, 0/80 and 1/80; no convincing claim about success rate follows.

## Planned: real enzyme benchmark

Before looking at real outcomes, fix:

- one task, reference provenance, catalytic identities, ligand atom mapping;
- generated vs fixed atoms, scoreable timestep range, missing-atom policy;
- constraint tolerances with a chemical/reference-based justification;
- none, best-of-N and FK configurations using the same conditioning;
- number of independent ensembles and sequences per backbone;
- fold evaluation settings and a common downstream selection policy;
- primary endpoint and diversity/quality constraints;
- generator calls, reward calls, downstream calls, GPU time and peak memory.

Report per-stage denominators: attempted -> generated -> MPNN -> prediction ->
valid atom mapping -> geometry pass -> combined pass. Failures stay visible.
Final all-atom geometry should be recomputed after downstream prediction; an
inference-time reward increase is not sufficient evidence.

Use direct distance/angle/dihedral errors and ligand-anchor RMSD where supported.
Report scRMSD/pLDDT separately. A free Kabsch fit of catalytic atoms can hide
their incorrect position relative to the ligand. Group uncertainty by independent
run, and report unique roots/structural diversity alongside top scores.

An initial particle count such as 4 is a smoke-test setting, not a validated
production choice. Set the experiment size after measuring runtime and memory.
