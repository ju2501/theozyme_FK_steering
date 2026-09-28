# CPU core method and contracts

## Geometry reward

For each constraint j, let e_j be measured minus target (wrapped to [-180,180)
for dihedrals). With tolerance t_j and scale s_j:

`v_j = max(0, abs(e_j) - t_j) / s_j`

`r(x) = -sum_j w_j v_j^2 / sum_j w_j`

Only constraints containing generated atoms enter r. All constraints enter the
diagnostic pass criterion. All fixed-only tasks are rejected for steering.
The squared hinge gives zero penalty in the tolerance band and avoids the flat
underflow of exponentiating a poor raw geometry score before FK. This version is
NumPy-only and does not implement gradients.

The toy task uses distances only and has mirror-related solutions; this is an
algorithm fixture, not a complete theozyme. Real chemistry needs reviewed atom
roles, angle/dihedral constraints and a meaningful ligand-relative frame.

Missing atoms, nonfinite coordinates, and undefined angles return invalid
scores (`-inf` internally, JSON null). A real adapter must treat a missing
representation as a mapping failure or define a justified reconstruction; it
must not substitute zeros or pretend that all sidechains are available.

## Sequential Monte Carlo

Core step indices ascend 1..S regardless of upstream's time convention. At each
scoring checkpoint c, define `phi_c = lambda * reward(prediction_c)` and set
`log_w += phi_c - phi_previous`, starting with `phi_previous = 0`.

Weights are normalized with max-subtracted exponentiation. ESS is
`1 / sum(normalized_weight**2)`. If ESS < fraction*N, systematic resampling
copies states, potential history and root ancestry; log weights then reset.
Otherwise weights persist to the next checkpoint. Subtracting their common
maximum changes no probabilities. Potential differences telescope between
checkpoints, including skipped intermediate steps.

Terminal scoring always occurs. For positive lambda FK, terminal resampling
always occurs, so the returned ensemble is unweighted. Omitting that final
selection would require exposing and using terminal importance weights.
`start_fraction/end_fraction` apply only to INTERMEDIATE selection. They do not
turn off the terminal correction. This convention is part of the experiment.

At finite N the returned samples are an approximation and correlated through
ancestry. Intermediate potentials guide allocation; they do not make the
finite-particle procedure exact. Killing an invalid lineage also restricts the
path population to trajectories valid at scored checkpoints. A previously
invalid lineage is never resurrected just because a later score is finite.
All invalid particles cause an explicit error.

Only difference potential is supported. Max/sum alternatives need their own
target/boundary accounting and must not be silently described as equivalent.

## RNG and state

NumPy SeedSequence creates one RNG per particle slot plus one selection RNG.
Cloned children use the distinct slot RNGs for subsequent transitions. A sampler
must actually consume the provided RNG; GPU adapters need explicit mapping to
Torch generators/noise sources. Independent seeds alone cannot diversify a
deterministic forward map. Model caches, self-conditioning, masks, assignments,
coordinates and any stochastic reward state are part of the clone contract.

## Alignment

Kabsch returns a proper rotation (determinant +1), translation and RMSD.
`anchored_rmsd` fits only anchor atoms and evaluates query atoms without a second
fit. Three non-collinear anchors are required. Equivalent ligand atoms and
sidechain symmetries are not automatically resolved.

Active-site-only alignment is useful for local shape, but can hide displacement
relative to ligand or scaffold. Always specify the fitted and evaluated atoms.
