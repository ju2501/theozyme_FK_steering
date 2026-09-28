# Runtime status

No SIF, model weights or GPU runtime was available during this CPU build.

| Item | Status |
|---|---|
| RFdiffusion2 upstream SHA | To be captured from actual checkout |
| RFdiffusion2 SIF source/hash | To be captured by probe |
| RFdiffusion2 weight hash | To be captured before experiment |
| LigandMPNN runtime | Not configured |
| AF2 runtime | Not configured; separate downstream variant |
| Colab native profile | Not verified |

Official references: [setup](https://rosettacommons.github.io/RFdiffusion2/readme_link.html),
[environment alternatives](https://rosettacommons.github.io/RFdiffusion2/installation.html).

The CPU source is intended to be bind-mounted read-only. Preserve the upstream
PYTHONPATH along with the steer source path. Model environments and the CPU
development environment are separate; do not use the host Python accidentally.
