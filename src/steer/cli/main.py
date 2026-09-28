import argparse
import json
from pathlib import Path
from steer.io.config import load_task
from steer.io.pdb import read_pdb, select_atoms
from steer.rewards.theozyme_reward import TheozymeReward


def main(argv=None):
    parser = argparse.ArgumentParser(description="Theozyme steering research core (CPU MVP)")
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score", help="Score mapped PDB atoms; invalid structures exit 2")
    score.add_argument("--task", required=True)
    score.add_argument("--pdb", required=True)
    demo = commands.add_parser("demo", help="Synthetic stochastic geometry benchmark, NOT RFdiffusion2")
    demo.add_argument("--config", required=True)
    demo.add_argument("--out", required=True)
    demo.add_argument("--resume", action="store_true")
    demo.add_argument("--plot", action="store_true")
    report = commands.add_parser("report", help="Plot an existing synthetic benchmark")
    report.add_argument("--out", required=True)
    report.add_argument("--destination")
    commands.add_parser("rfd2", help="Reports the real adapter's unsupported status")
    args = parser.parse_args(argv)
    try:
        if args.command == "score":
            task = load_task(args.task)
            result = TheozymeReward(task)(select_atoms(read_pdb(args.pdb), task.atoms))
            print(json.dumps(result.to_dict(), indent=2, allow_nan=False))
            return 0 if result.valid else 2
        if args.command == "demo":
            from .benchmark import benchmark, plot_report
            summary = benchmark(args.config, args.out, args.resume)
            if args.plot:
                plot_report(args.out)
            print(json.dumps(summary, indent=2, allow_nan=False))
        elif args.command == "report":
            from .benchmark import plot_report
            print(plot_report(args.out, args.destination))
        else:
            from steer.hooks.rfd2_hook import RFdiffusion2Sampler
            RFdiffusion2Sampler()
    except (ValueError, KeyError, TypeError, OSError, NotImplementedError) as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
