"""Save a finished game and play the same one back from its seed."""

from __future__ import annotations

import json
from pathlib import Path

from prsi.game import Game, IllegalAction


RECORD_VERSIONS = (1, 2)
# Games saved before ``opening_hand`` was written were dealt four cards each.
LEGACY_OPENING_HAND = 4


def save_record(
    game: Game,
    path,
    names: list[str] | None = None,
    seats: list[dict] | None = None,
    mode: int | None = None,
    result: dict | None = None,
) -> Path:
    """Write a finished game.

    Version 2 adds ``seats``: who sat where, e.g. ``{"kind": "human", "name": "Ada"}``
    or ``{"kind": "bot", "engine": "guided", "network": "gen_003"}``.
    ``result`` overrides ``game.result`` when the hand is saved without being scored,
    as when the host resets mid-hand.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(names) if names is not None else ["Player 1", "Player 2"]
    data = {
        "version": 2,
        "seed": game.seed,
        "starting_player": game.starting_player,
        "opening_hand": game.opening_hand,
        "actions": list(game.actions),
        "result": game.result if result is None else result,
        "names": names,
        "seats": seats if seats is not None else [{"kind": "human", "name": name} for name in names],
    }
    if mode is not None:
        data["mode"] = mode
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def load_record(path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") not in RECORD_VERSIONS:
        raise ValueError("This game record is from a different version of the rules.")
    if "seats" not in data:
        names = data.get("names") or ["Player 1", "Player 2"]
        data["seats"] = [{"kind": "human", "name": name} for name in names]
    return data


def opening_hand_of(record: dict) -> int:
    """Cards dealt to each seat. A record without the field was dealt 4."""
    if "opening_hand" not in record or record["opening_hand"] is None:
        return LEGACY_OPENING_HAND
    return int(record["opening_hand"])


def replay(record: dict) -> Game:
    """Play ``record`` from scratch. Same seed, same moves, same game."""
    game = Game(record["seed"], record["starting_player"], opening_hand_of(record))
    for index, action in enumerate(record["actions"]):
        try:
            game.step(action)
        except IllegalAction as exc:
            raise IllegalAction(f"Replay failed on move {index} (action {action}).") from exc
    return game


def format_event(event: dict, names: tuple[str, ...] | list[str] = ("Player 1", "Player 2")) -> str:
    """One line of the public log, using seat names rather than You/Opponent."""

    def who(player: int | None) -> str:
        if player is None:
            return "Someone"
        if 0 <= player < len(names):
            return names[player]
        return f"Player {player + 1}"

    kind = event.get("kind")
    if kind == "opening":
        return f"Opening card: {event['card']['name']}."
    if kind == "play":
        line = f"{who(event['player'])} played {event['card']['name']}"
        if event.get("suit_name"):
            line += f" and named {event['suit_name']}"
        return line + "."
    if kind == "draw":
        count = event["count"]
        noun = "card" if count == 1 else "cards"
        if event.get("penalty"):
            return f"{who(event['player'])} took {count} {noun}."
        if count == 1:
            return f"{who(event['player'])} drew a card."
        return f"{who(event['player'])} drew {count} {noun}."
    if kind == "stand":
        return f"{who(event['player'])} stood."
    if kind == "pass":
        line = f"{who(event['player'])} passed"
        if event.get("forgiven"):
            line += " — no cards left to take"
        return line + "."
    if kind == "shuffle":
        return "The discard pile was shuffled into the draw pile."
    if kind == "game_over":
        winner = event.get("winner")
        if winner is None:
            return "The game is a draw."
        return f"{who(winner)} wins."
    return str(kind)
