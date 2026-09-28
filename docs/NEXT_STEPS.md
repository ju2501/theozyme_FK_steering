# Next working session

The original documents have now become a runnable CPU implementation. The next
dependency is an actual RFdiffusion2 runtime and source checkout.

| Order | Concrete work | Evidence needed |
|---|---|---|
| 1 | Run tests and demo on Crostini | Test output; install errors if any |
| 2 | Probe lab or Colab runtime | runtime_probe.json; actual GPU and upstream SHA |
| 3 | Run unchanged RFdiffusion2 active-site example | PDB/TRB, full log, resolved config |
| 4 | Inspect and implement full sampler state adapter | Actual source; clone and zero-steering checks |
| 5 | Validate real catalytic reward | Atom mapping; nonconstant intermediate score traces |
| 6 | Run a small FK smoke test | ESS, ancestry, memory/runtime, sibling divergence |
| 7 | Add common LigandMPNN + chosen prediction pipeline | Fixed catalytic identities and tracked residue mapping |
| 8 | Run matched-budget independent comparisons | none/best-of-N/FK stage metrics and failure counts |
| 9 | Update README and publish | Real result provenance, chosen license, target GitHub repo |

If remote lab access works, prefer the already-tested lab runtime to spending
the holiday rebuilding a complex Colab environment. If only Colab is available,
complete its baseline test before planning the experiment size.

Holiday delivery target: a reproducible small study with honest results. Positive
FK performance, reliable enzyme activity, and broad generality cannot be promised.
Gradient guidance, multiple reactions, PyRosetta reward loops and automated SLURM
submission can follow once the single-task path works.
