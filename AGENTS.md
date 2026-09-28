# Working on theozyme-steer

This repository is currently a CPU research core, not a working RFdiffusion2 steering pipeline.
Read README.md and the latest appendix of DECISIONS.md before making changes.

- Core imports may depend on the standard library and NumPy only. Reporting dependencies are lazy imports.
- Run `python -m unittest discover -s tests -v` after core changes. Pytest can also collect these tests.
- Run `python -m steer.cli.main demo --config configs/runs/toy.json --out runs/<new-name>` for the synthetic integration check.
- Never label synthetic results as RFdiffusion2, enzyme activity, or GPU validation.
- Keep fixed vs generated atom roles explicit; verify nonconstant reward on real predicted structures.
- Preserve all sampler caches and self-conditioning state during resampling. Coordinates alone are insufficient.
- Keep adaptive SMC weights between selection events. Final weighting is mandatory for the declared terminal objective.
- Compare unguided, best-of-N, and FK using recorded compute budgets and independent ensemble repetitions.
- RFdiffusion2 integration is blocked on a pinned upstream source and runtime probe. The current adapter raises NotImplementedError intentionally.
- Append design changes to DECISIONS.md. Do not replace previous decisions silently.
- Do not commit weights, SIF files, credentials, local machine profiles, or large run folders.
