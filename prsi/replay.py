"""Print a saved game, move by move.

    python -m prsi.replay games\\20261026T120000Z_123.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from prsi.game import IllegalAction
from prsi.record import format_event, load_record, replay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay a saved Prší game.")
    parser.add_argument("record", help="Path to a game JSON file.")
    args = parser.parse_args(argv)
    path = Path(args.record)
    if not path.is_file():
        print(f"No such file: {path}", file=sys.stderr)
        return 1
    try:
        record = load_record(path)
        game = replay(record)
    except (OSError, ValueError, IllegalAction, KeyError, TypeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    names = tuple(record.get("names") or ["Player 1", "Player 2"])
    for event in game.log:
        print(format_event(event, names))
    result = game.result
    if result is None:
        print("The game is not over.")
    elif result["winner"] is None:
        print("Result: draw.")
    else:
        reason = "last card" if result["reason"] == "empty_hand" else "neither player could move"
        print(f"Result: {names[result['winner']]} wins ({reason}).")
    print(f"Moves: {len(game.actions)}")
    if record.get("result") != game.result:
        print("Recorded result does not match the replay.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
