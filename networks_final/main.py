"""

Modes
-----_________----------------_________________-----------------___________
  visualize   Interactive Pygame window
  experiment  Run 100 paired trials and generate Matplotlib plots
  both        Run visualisation first, then the experiment
_-----______----_______________-----------______________-------------________
Examples
---------______________________-------------------_______________----------------
  python main.py visualize
  python main.py experiment --trials 100 --sensitivity
  python main.py experiment --trials 20
  python main.py visualize --n-nodes 30 --beta 500.0
-___----------------____________________---------------___________------------_____
"""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("--n-nodes",   type=int,   default=30,    help="Number of MANET nodes")
    parent.add_argument("--max-steps", type=int,   default=1000,   help="Max timesteps per episode")
    parent.add_argument("--alpha",     type=float, default=1.0,   help="Hop-weight in energy-aware cost")
    parent.add_argument("--beta",      type=float, default=500.0,  help="Energy-weight in energy-aware cost")
    parent.add_argument("--tx-range",  type=float, default=150.0, help="Transmission range (m)")
    parent.add_argument("--n-flows",   type=int,   default=5,     help="Number of traffic flows")
    parent.add_argument("--speed",     type=float, default=2.0,   help="Node speed (m/step)")

    parser = argparse.ArgumentParser(
        description="MANET Energy-Aware Routing Simulation — COMP 4203",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # visualize subcommand
    vis_p = sub.add_parser("visualize", parents=[parent],
                           help="Live Pygame visualisation")
    vis_p.add_argument("--seed", type=int, default=0, help="Initial random seed")

    # experiment subcommand
    exp_p = sub.add_parser("experiment", parents=[parent],
                           help="Run 100 paired trials with statistical analysis")
    exp_p.add_argument("--trials",      type=int,  default=100,     help="Number of trial pairs")
    exp_p.add_argument("--out-dir",     type=str,  default="results", help="Output directory for plots")
    exp_p.add_argument("--sensitivity", action="store_true",          help="Run alpha/beta sensitivity grid")

    # both 
    both_p = sub.add_parser("both", parents=[parent],
                             help="Run visualisation, then experiment")
    both_p.add_argument("--seed",        type=int,  default=0)
    both_p.add_argument("--trials",      type=int,  default=100)
    both_p.add_argument("--out-dir",     type=str,  default="results")
    both_p.add_argument("--sensitivity", action="store_true")

    return parser


def env_kwargs(args) -> dict:
    return dict(
        n_nodes   = args.n_nodes,
        max_steps = args.max_steps,
        alpha     = args.alpha,
        beta      = args.beta,
        tx_range  = args.tx_range,
        n_flows   = args.n_flows,
        node_speed = args.speed,
    )


def main():
    parser = build_parser()
    args   = parser.parse_args()
    kw     = env_kwargs(args)

    if args.mode in ("visualize", "both"):
        print("\n[MANET] Launching Pygame visualisation …")
        print("  Controls: SPACE=pause  TAB=toggle algorithm  R=reset  +/-=speed  Q=quit\n")
        from visualize import run_live
        run_live(env_kwargs=kw, seed=args.seed)

    if args.mode in ("experiment", "both"):
        print("\n[MANET] Launching experiment runner …")
        from experiment import run_experiment
        run_experiment(
            n_trials        = args.trials,
            run_sensitivity = args.sensitivity,
            out_dir         = args.out_dir,
            env_kwargs      = kw,
        )


if __name__ == "__main__":
    main()
