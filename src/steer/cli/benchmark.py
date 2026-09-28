"""Reproducible synthetic benchmark. Never presented as enzyme evidence."""
from dataclasses import asdict
from pathlib import Path
import csv
import hashlib
import platform
import time
import numpy as np
from steer import __version__
from steer.io.config import load_json, load_task, atomic_json, file_hash
from steer.particles.fk_steering import FKConfig, run_particles
from steer.hooks.toy import ToySampler
from steer.rewards.theozyme_reward import TheozymeReward


def source_hash():
    root = Path(__file__).parents[1]
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        h.update(str(path.relative_to(root)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def _spread(states):
    if len(states) < 2:
        return 0.0
    x = np.asarray(states)
    return float(np.linalg.norm(x[:, None] - x[None, :], axis=-1)[np.triu_indices(len(x), 1)].mean())


def benchmark(config_path, out, resume=False):
    config_path, out = Path(config_path), Path(out)
    settings = load_json(config_path)
    if set(settings) != {"task", "replicates", "seed", "select_k", "fk"}:
        raise ValueError("Unknown or missing benchmark config keys")
    cfg = FKConfig(**settings["fk"])
    for key in ("replicates", "select_k", "seed"):
        if type(settings[key]) is not int or settings[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid {key}")
    if settings["select_k"] > cfg.num_particles:
        raise ValueError("select_k cannot exceed num_particles")
    task_path = config_path.parent / settings["task"]
    task = load_task(task_path)
    task.validate_for_steering()
    if set(task.atoms) != {"a", "b", "c", "d"}:
        raise ValueError("Toy sampler requires synthetic labels a,b,c,d")
    sampler, scorer = ToySampler(), TheozymeReward(task)
    identity = {"schema": 1, "kind": "synthetic_geometry_not_enzyme", "config": settings,
                "task_sha256": file_hash(task_path), "source_sha256": source_hash(),
                "package_version": __version__, "python": platform.python_version(), "numpy": np.__version__}
    manifest_path = out / "manifest.json"
    if resume:
        if not manifest_path.exists() or load_json(manifest_path) != identity:
            raise ValueError("Resume identity mismatch (config/task/source/runtime), or missing manifest")
    else:
        out.mkdir(parents=True, exist_ok=False)
        atomic_json(manifest_path, identity)
        atomic_json(out / "task.json", load_json(task_path))
    (out / "replicates").mkdir(exist_ok=True)
    all_rows, replicate_summaries = [], []
    for rep in range(settings["replicates"]):
        path = out / "replicates" / f"{rep:04d}.json"
        if resume and path.exists():
            record = load_json(path)
        else:
            seed = settings["seed"] + rep
            outputs, elapsed = {}, {}
            for mode in ("none", "fk"):
                start = time.perf_counter()
                outputs[mode] = run_particles(sampler, lambda x: scorer(x).value, cfg, seed, mode)
                elapsed[mode] = time.perf_counter() - start
            base, fk = outputs["none"], outputs["fk"]
            k = settings["select_k"]
            # Selection RNG is independent of model and resampling streams.
            rng = np.random.default_rng(np.random.SeedSequence([seed, 314159]))
            uniform_indices = rng.choice(cfg.num_particles, k, replace=False)
            selected = {"unguided": (base, uniform_indices),
                        "best_of_n": (base, np.argsort(-base.rewards, kind="stable")[:k]),
                        "fk": (fk, rng.choice(cfg.num_particles, k, replace=False))}
            rows, summaries = [], []
            for arm, (result, indices) in selected.items():
                states = [result.states[int(i)] for i in indices]
                scores = [scorer(sampler.predict(s)) for s in states]
                for slot, (i, state, score) in enumerate(zip(indices, states, scores)):
                    rows.append({"replicate": rep, "seed": seed, "arm": arm, "slot": slot,
                                 "particle_slot": int(i), "root_ancestor": int(result.ancestry[i]),
                                 "reward": score.value if score.valid else None,
                                 "valid": score.valid, "passed": score.passed,
                                 "x": float(state[0]), "y": float(state[1]), "z": float(state[2])})
                summaries.append({"replicate": rep, "arm": arm,
                                  "mean_reward": float(np.mean([s.value for s in scores])),
                                  "pass_fraction": float(np.mean([s.passed for s in scores])),
                                  "pairwise_coordinate_spread": _spread(states),
                                  "unique_roots_selected": int(len(set(result.ancestry[indices]))),
                                  "model_step_calls": result.model_step_calls,
                                  "selection_reward_calls": result.reward_calls,
                                  "sampler_elapsed_seconds": elapsed["fk" if arm == "fk" else "none"]})
            record = {"replicate": rep, "rows": rows, "summaries": summaries,
                      "unguided_pool": [s.tolist() for s in base.states],
                      "fk_pool": [s.tolist() for s in fk.states], "fk_history": fk.history}
            atomic_json(path, record)
        all_rows.extend(record["rows"])
        replicate_summaries.extend(record["summaries"])
    for name, rows in (("selected.csv", all_rows), ("replicate_metrics.csv", replicate_summaries)):
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {}
    for arm in ("unguided", "best_of_n", "fk"):
        subset = [x for x in replicate_summaries if x["arm"] == arm]
        means = np.array([x["mean_reward"] for x in subset])
        summary[arm] = {"independent_replicates": len(subset), "mean_reward": float(means.mean()),
                        "sd_across_replicates": float(means.std(ddof=1)) if len(means) > 1 else None,
                        "pass_fraction": float(np.mean([x["pass_fraction"] for x in subset])),
                        "mean_coordinate_spread": float(np.mean([x["pairwise_coordinate_spread"] for x in subset])),
                        "model_step_calls_per_replicate": cfg.num_particles * cfg.steps}
    # Paired uncertainty uses independent ensembles, never correlated particles.
    fk_means = np.array([x["mean_reward"] for x in replicate_summaries if x["arm"] == "fk"])
    best_means = np.array([x["mean_reward"] for x in replicate_summaries if x["arm"] == "best_of_n"])
    delta = fk_means - best_means
    ci = None
    if len(delta) > 1:
        bootstrap_rng = np.random.default_rng(987654)
        boot = bootstrap_rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
        ci = [float(x) for x in np.quantile(boot, [0.025, 0.975])]
    summary["paired_fk_minus_best_of_n"] = {"mean_reward_difference": float(delta.mean()),
                                           "bootstrap_95_percent_interval": ci}
    atomic_json(out / "summary.json", summary)
    return summary


def plot_report(out, destination=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out)
    with (out / "replicate_metrics.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    arms = ("unguided", "best_of_n", "fk")
    colors = ("#778DA9", "#E09F3E", "#007F73")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
    for ax, metric, title, ylabel in zip(axes, ("mean_reward", "pairwise_coordinate_spread"),
                                        ("Constraint reward (higher is better)", "Within-ensemble coordinate spread"),
                                        ("Mean selected reward", "Mean pairwise distance (toy units)")):
        for i, (arm, color) in enumerate(zip(arms, colors)):
            values = np.array([float(r[metric]) for r in rows if r["arm"] == arm])
            x = i + np.linspace(-0.12, 0.12, len(values))
            ax.scatter(x, values, color=color, alpha=0.65, s=22)
            ax.plot([i - .22, i + .22], [values.mean()] * 2, color="black", lw=2)
        ax.set_xticks(range(3), ["Unguided", "Best-of-N", "FK"])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Synthetic geometry demo — NOT RFdiffusion2 or enzyme results", fontsize=13, weight="bold")
    fig.supxlabel("One dot = one independent ensemble. Equal sampler calls; reward cost differs.", fontsize=9)
    path = Path(destination) if destination else out / "comparison.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
