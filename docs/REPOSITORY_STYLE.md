# Repository presentation direction

Reviewed on 2026-09-27. These repositories are presentation references, not claims
that this CPU package reproduces their biological pipelines.

| Reference | Pattern to adopt | Application here |
|---|---|---|
| [UvA tutorials](https://github.com/phlippe/uvadlc_notebooks) | Theory alongside runnable notebooks; CPU/Colab entry points | Short geometry, FK and comparison tutorials, each with expected outputs |
| [Heme binder pipeline](https://github.com/ikalvet/heme_binder_diffusion) | A visible sequence of design/prediction/analysis stages | Explicit stage inputs, outputs, runtime requirements and implemented status |
| [RFdiffusion](https://github.com/RosettaCommons/RFdiffusion) | Installation, concrete command examples, parameter explanations and output guidance | Copyable CPU quickstart first, then validated GPU recipes |

The heme example uses RFdiffusionAA; RFdiffusion is distinct from RFdiffusion2.
Their commands and model environments are not interchangeable.

## First public-facing version

1. A short research question and accurate implementation status.
2. A working CPU installation and demo that a reader can run.
3. One clearly labelled synthetic result figure plus exact reproduction config.
4. Method documentation explaining reward and resampling with failure cases.
5. CPU tests and GitHub Actions; show badges only after the actual workflow runs.
6. A roadmap separating real RFdiffusion2 integration from implemented CPU work.

## After the Chromebook check

Create tutorial notebooks that import the tested library instead of duplicating
its algorithms. Suggested order: geometry -> reward -> FK particle ancestry ->
matched-budget comparison. Only add an Open-in-Colab link after the repository
URL and notebook runtime have been verified. Add a real enzyme pipeline notebook
when the GPU adapter and downstream stages work.

Prepare an English overview for wider readers and retain Korean setup notes.
Select a license for original code before presenting an open-source release.
Keep raw runs out of Git, but keep all reproducibility configs tracked.
