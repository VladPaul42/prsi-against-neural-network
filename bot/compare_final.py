"""Does the network-guided search beat both of its parts?

Plays the final bot (network-guided ISMCTS) against plain ISMCTS and against
the network alone, every bot with the same time per move, on mirrored deals.

    python -m bot.compare_final --network runs/league/gen_001 --games 500 --seconds 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from bot.league import best_generation_dir
from bot.match import run_match


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default=None, help="Defaults to the best generation in runs/league.")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    folder = Path(args.network) if args.network else best_generation_dir()
    if folder is None:
        raise SystemExit("No trained network yet.")
    label = folder.name
    guided = {"kind": "guided", "path": str(folder), "seconds": args.seconds, "label": label,
              "name": f"guided-{label}-{args.seconds}s"}
    plain = {"kind": "ismcts", "seconds": args.seconds, "name": f"ismcts-{args.seconds}s"}
    alone = {"kind": "ppo", "path": str(folder), "greedy": True, "label": label, "name": f"ppo-{label}"}
    results = []
    for opponent in (plain, alone):
        summary = run_match(guided, opponent, args.games, workers=args.workers)
        results.append(summary)
    print()
    for summary in results:
        print(f"{summary['a']} vs {summary['b']}: {summary['a_score']:.1%} "
              f"({summary['a_wins']} wins, {summary['b_wins']} losses, {summary['draws']} draws)")


if __name__ == "__main__":
    main()
