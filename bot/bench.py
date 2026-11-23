"""Random-play speed of both engines.

    python -m bot.bench --games 2000
"""

from __future__ import annotations

import argparse
import random
import time

from bot.fastgame import REASON_LIMIT, FastGame
from prsi.game import Game


def bench_reference(games: int, seed: int = 42) -> dict:
    master = random.Random(seed)
    actions = 0
    start = time.perf_counter()
    for _ in range(games):
        game = Game(master.randrange(2**63), master.randrange(2))
        choice = master.choice
        while not game.is_over:
            game.step(choice(game.legal_actions()))
        actions += len(game.actions)
    elapsed = time.perf_counter() - start
    return {"games_per_s": games / elapsed, "actions_per_game": actions / games}


def bench_fast(games: int, seed: int = 42, max_actions: int | None = 300) -> dict:
    master = random.Random(seed)
    actions = 0
    limited = 0
    lengths = []
    start = time.perf_counter()
    for _ in range(games):
        game = FastGame(master.randrange(2**63), master.randrange(2), max_actions=max_actions)
        choice = master.choice
        while not game.over:
            game.step(choice(game.legal_actions()))
        actions += game.n_actions
        lengths.append(game.n_actions)
        if game.reason == REASON_LIMIT:
            limited += 1
    elapsed = time.perf_counter() - start
    lengths.sort()
    return {
        "games_per_s": games / elapsed,
        "actions_per_s": actions / elapsed,
        "actions_per_game": actions / games,
        "median_length": lengths[len(lengths) // 2],
        "p99_length": lengths[int(len(lengths) * 0.99)],
        "longest": lengths[-1],
        "hit_limit": limited,
        "hit_limit_pct": 100.0 * limited / games,
    }


def bench_clone(copies: int = 200_000) -> float:
    game = FastGame(42, 0)
    rng = random.Random(0)
    for _ in range(10):
        game.step(rng.choice(game.legal_actions()))
    start = time.perf_counter()
    for _ in range(copies):
        game.clone(rng)
    return copies / (time.perf_counter() - start)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Random-play speed of both engines.")
    parser.add_argument("--games", type=int, default=2000)
    parser.add_argument("--limit-games", type=int, default=100_000,
                        help="Random games used to measure how often the 300-action cap is hit.")
    args = parser.parse_args(argv)
    ref = bench_reference(args.games)
    fast = bench_fast(args.games * 5)
    print(f"Reference engine: {ref['games_per_s']:8.0f} games/s "
          f"({ref['actions_per_game']:.1f} actions per game)")
    print(f"Fast engine:      {fast['games_per_s']:8.0f} games/s "
          f"({fast['actions_per_s']:.0f} actions/s)")
    print(f"Speed-up:         {fast['games_per_s'] / ref['games_per_s']:8.1f}x")
    print(f"Clones per second (shared rng): {bench_clone():.0f}")
    capped = bench_fast(args.limit_games, seed=42)
    print(
        f"300-action cap over {args.limit_games} random games: hit {capped['hit_limit']} times "
        f"({capped['hit_limit_pct']:.3f}%); median length {capped['median_length']}, "
        f"99th percentile {capped['p99_length']}, longest {capped['longest']}"
    )


if __name__ == "__main__":
    main()
