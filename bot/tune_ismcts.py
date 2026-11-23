"""Small experiments for the ISMCTS settings, on mirrored deals.

    python -m bot.tune_ismcts --games 200 --workers 6
"""

from __future__ import annotations

import argparse
import time

from bot.match import run_match

BASE = {"kind": "ismcts", "iterations": 1000, "c": 0.7, "draw_prob": 0.0}


def experiments() -> list[tuple[str, dict, dict]]:
    def variant(name, **changes):
        return {**BASE, **changes, "name": name}

    base = variant("base: 1000it c0.7 play-first")
    return [
        ("strength vs random", base, {"kind": "random"}),
        ("rollout: uniform random", base, variant("uniform rollout", draw_prob=None)),
        ("rollout: draw 10% of the time", base, variant("draw 10%", draw_prob=0.1)),
        ("exploration c=0.4", base, variant("c0.4", c=0.4)),
        ("exploration c=1.0", base, variant("c1.0", c=1.0)),
        ("budget 250 imagined games", base, variant("250it", iterations=250)),
        ("budget 4000 imagined games", base, variant("4000it", iterations=4000)),
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--only", type=int, nargs="*", help="Run only these experiment numbers.")
    args = parser.parse_args(argv)
    for index, (title, a, b) in enumerate(experiments()):
        if args.only and index not in args.only:
            continue
        start = time.perf_counter()
        summary = run_match(a, b, args.games, workers=args.workers, progress=False)
        print(
            f"[{index}] {title}: base scores {summary['a_score']:.1%} "
            f"({summary['a_wins']}-{summary['b_wins']}-{summary['draws']}), "
            f"avg {summary['avg_length']} actions, {time.perf_counter() - start:.0f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
