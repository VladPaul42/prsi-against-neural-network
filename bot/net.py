"""The policy/value network, and a bot that plays with it alone.

Training uses PyTorch. Playing uses a NumPy copy of the same weights
(``weights.npz``), which is faster for one position at a time and lets worker
processes skip importing torch.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from bot.agents import Agent, PublicState
from bot.encode import INPUT_SIZE, encode
from bot.fastgame import NUM_ACTIONS

VALUE_HIDDEN = 64
ROOT = Path(__file__).resolve().parent.parent


def build_model(hidden: int = 256, layers: int = 2):
    """A small MLP with a policy head (45 logits) and a value head (tanh)."""
    import torch
    from torch import nn

    class PolicyValueNet(nn.Module):
        def __init__(self):
            super().__init__()
            body = []
            size = INPUT_SIZE
            for _ in range(layers):
                body += [nn.Linear(size, hidden), nn.ReLU()]
                size = hidden
            self.body = nn.Sequential(*body)
            self.policy = nn.Linear(hidden, NUM_ACTIONS)
            self.value_hidden = nn.Linear(hidden, VALUE_HIDDEN)
            self.value_out = nn.Linear(VALUE_HIDDEN, 1)
            nn.init.orthogonal_(self.policy.weight, gain=0.01)
            nn.init.zeros_(self.policy.bias)

        def forward(self, x, mask):
            h = self.body(x)
            logits = self.policy(h).masked_fill(~mask, -1e9)
            value = torch.tanh(self.value_out(torch.relu(self.value_hidden(h)))).squeeze(-1)
            return logits, value

    model = PolicyValueNet()
    model.hidden = hidden
    model.layers = layers
    return model


def model_to_arrays(model) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy().copy() for key, value in model.state_dict().items()}


class NumpyNet:
    """Inference-only copy of ``PolicyValueNet``."""

    def __init__(self, arrays: dict[str, np.ndarray]):
        self.body = []
        index = 0
        while f"body.{index}.weight" in arrays:
            self.body.append(
                (np.ascontiguousarray(arrays[f"body.{index}.weight"].T, dtype=np.float32),
                 arrays[f"body.{index}.bias"].astype(np.float32))
            )
            index += 2
        self.pw = np.ascontiguousarray(arrays["policy.weight"].T, dtype=np.float32)
        self.pb = arrays["policy.bias"].astype(np.float32)
        self.vw1 = np.ascontiguousarray(arrays["value_hidden.weight"].T, dtype=np.float32)
        self.vb1 = arrays["value_hidden.bias"].astype(np.float32)
        self.vw2 = np.ascontiguousarray(arrays["value_out.weight"].T, dtype=np.float32)
        self.vb2 = arrays["value_out.bias"].astype(np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Raw logits and value (-1..1). ``x`` is one row or a batch."""
        h = x
        for weight, bias in self.body:
            h = h @ weight + bias
            np.maximum(h, 0.0, out=h)
        logits = h @ self.pw + self.pb
        v = h @ self.vw1 + self.vb1
        np.maximum(v, 0.0, out=v)
        value = np.tanh(v @ self.vw2 + self.vb2)[..., 0]
        return logits, value

    def policy_value(self, x: np.ndarray, legal: list[int]) -> tuple[np.ndarray, float]:
        """Probabilities over ``legal`` (same order) and the value for the player to move."""
        logits, value = self.forward(x)
        chosen = logits[legal].astype(np.float64)
        chosen -= chosen.max()
        probs = np.exp(chosen)
        probs /= probs.sum()
        return probs, float(value)

    def value(self, x: np.ndarray) -> float:
        h = x
        for weight, bias in self.body:
            h = h @ weight + bias
            np.maximum(h, 0.0, out=h)
        v = h @ self.vw1 + self.vb1
        np.maximum(v, 0.0, out=v)
        return float(np.tanh(v @ self.vw2 + self.vb2)[0])


_NET_CACHE: dict[str, NumpyNet] = {}


def resolve_weights(path: str | Path) -> Path:
    """Accept a generation folder, ``model.pt`` or ``weights.npz``.

    Relative paths are taken from the project folder.
    """
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    if path.is_dir():
        return path / "weights.npz"
    if path.suffix == ".pt":
        return path.with_name("weights.npz")
    return path


def load_numpy_net(path: str | Path) -> NumpyNet:
    weights = resolve_weights(path)
    key = str(weights.resolve())
    net = _NET_CACHE.get(key)
    if net is None:
        with np.load(weights) as data:
            net = NumpyNet({name: data[name] for name in data.files})
        _NET_CACHE[key] = net
    return net


def save_network(model, folder: str | Path, meta: dict | None = None) -> Path:
    """Write ``model.pt`` (PyTorch), ``weights.npz`` (NumPy) and ``meta.json``."""
    import torch

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "hidden": model.hidden, "layers": model.layers},
        folder / "model.pt",
    )
    np.savez(folder / "weights.npz", **model_to_arrays(model))
    if meta is not None:
        (folder / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return folder


def load_model(folder: str | Path):
    import torch

    folder = Path(folder)
    path = folder / "model.pt" if folder.is_dir() else folder
    data = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(data["hidden"], data["layers"])
    model.load_state_dict(data["state_dict"])
    return model


class PolicyAgent(Agent):
    """Plays the network's move: the most likely one (``greedy``) or a sample."""

    name = "ppo"

    def __init__(self, net: NumpyNet, greedy: bool = True, seed: int = 0):
        self.net = net
        self.greedy = greedy
        self.rng = random.Random(seed)
        self._x = np.zeros(INPUT_SIZE, dtype=np.float32)

    @classmethod
    def from_spec(cls, spec: dict) -> PolicyAgent:
        return cls(load_numpy_net(spec["path"]), spec.get("greedy", True), spec.get("seed", 0))

    def new_game(self, seed: int) -> None:
        self.rng.seed(seed)

    def evaluate(self, state: PublicState) -> tuple[np.ndarray, float]:
        return self.net.policy_value(encode(state, self._x), state.legal)

    def act(self, state: PublicState) -> int:
        if len(state.legal) == 1:
            return state.legal[0]
        probs, _ = self.evaluate(state)
        if self.greedy:
            return state.legal[int(np.argmax(probs))]
        return state.legal[_sample(probs, self.rng.random())]

    def analyse(self, state: PublicState, progress=None) -> dict:
        probs, value = self.evaluate(state)
        best = int(np.argmax(probs))
        action = state.legal[best] if self.greedy else state.legal[_sample(probs, self.rng.random())]
        return {
            "action": action,
            "win_chance": (value + 1.0) / 2.0,
            "iterations": 0,
            "seconds": 0.0,
            "moves": [
                {"action": a, "visits": 0, "win_chance": None, "prior": float(p)}
                for a, p in zip(state.legal, probs)
            ],
        }


def _sample(probs: np.ndarray, u: float) -> int:
    total = 0.0
    for index, p in enumerate(probs):
        total += p
        if u < total:
            return index
    return len(probs) - 1
