# RFdiffusion2 integration gate

## Current state

`RFdiffusion2Sampler` deliberately raises NotImplementedError. No function name,
source line, checkpoint, commit SHA or GPU compatibility is claimed. During this
build, the official docs were accessible but requests for upstream Python source
were blocked. There is no GPU or user lab session attached to this workspace.

The [official documentation](https://rosettacommons.github.io/RFdiffusion2/readme_link.html)
describes predicted X1 in flow-matching trajectories and unindexed atomic motifs.
Those facts make a generic RFdiffusion x0 callback insufficient. Bind-mounting
this package only makes it importable; it does not insert a sampler hook.

## Required evidence before implementing the real adapter

1. Pin a full upstream SHA. Capture SIF hash and model weights identifiers.
2. Run the upstream active-site demo unchanged and retain PDB/TRB/log/config.
3. Inspect one complete sampler step and how it propagates time, self-conditioning,
   masks, motif mapping and sequence/index assignments.
4. Expose the predicted terminal structure before reward calculation. Identify
   which catalytic atoms are predictions and which are copied fixed inputs.
5. Record reward and each constraint over several particles and checkpoints.
   Fixed copies alone must not produce a claimed steering signal.
6. Test clone isolation and whether sibling continuations actually diverge.
   If propagation is deterministic, characterize early selection of initial
   trajectories and/or design a justified stochastic transition separately.
   Do not add arbitrary jitter and call it unchanged upstream sampling.
7. Run zero-steering equivalence against the unchanged upstream baseline using
   the same RNG/noise policy and conditioning. Record any floating point limits.

## Adapter API

- `initialize(rng)`: full independent mutable particle state.
- `advance(state, step, total_steps, rng)`: one real model transition.
- `predict(state)`: atom-name mapped terminal prediction used by reward.
- `clone(state)`: independent mutable state, including caches and assignments.

Share read-only model weights across particles. A sequential implementation can
trade runtime for lower GPU memory. Do not assume N particles require N copies
of the model. Tensor-to-NumPy transfer should extract only needed atoms at scored
checkpoints; avoid copying the whole trajectory to CPU every step.

If a terminal prediction requires an extra forward pass, count that pass and
its latency in the budget. The CPU toy's predict() has no model call and must
not be used as a real RFdiffusion2 cost estimate.

## Downstream

The upstream documented example links LigandMPNN and Chai-1. The user's AF2
evaluation is a separate pipeline variant. Preserve catalytic residue identities
and track their new residue numbers through LigandMPNN and prediction. Record
what ligand information the prediction tool actually consumes. Protein-only
refolding cannot alone establish ligand-relative catalytic accuracy.

The initial real task should come from a verified upstream example with usable
atom mapping. Do not invent a Ser-His-Asp configuration just to fill a filename.
