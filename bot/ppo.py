"""Hand-written PPO for the Prší network, CPU only.

The only reward is the final result: +1 win, -1 loss, 0 draw. Worker
processes play games between the learner (sampling from its policy) and one
opponent, and return the learner's decisions with GAE advantages already
computed. The main process does the gradient updates.

Workers never import torch: they use the NumPy copy of the weights.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field

import numpy as np

from bot.agents import PublicState, make_agent
from bot.encode import INPUT_SIZE, encode
from bot.fastgame import NUM_ACTIONS, FastGame
from bot.net import NumpyNet, load_numpy_net
from bot.tracker import Tracker


@dataclass
class PPOConfig:
    learning_rate: float = 3e-4
    clip: float = 0.2
    epochs: int = 4
    minibatch: int = 512
    gamma: float = 1.0
    gae_lambda: float = 0.95
    value_coef: float = 0.5
    # One coefficient per stage. Each stage lasts LeagueConfig.updates_per_gen updates.
    entropy_coef: list[float] = field(default_factory=lambda: [0.01])
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    hidden: int = 256
    layers: int = 2
    games_per_update: int = 256
    games_per_job: int = 8

    def __post_init__(self) -> None:
        if isinstance(self.entropy_coef, (int, float)):
            self.entropy_coef = [float(self.entropy_coef)]
        else:
            self.entropy_coef = [float(value) for value in self.entropy_coef]
        if not self.entropy_coef:
            raise ValueError("entropy_coef must contain at least one value")

    def entropy_at(self, update: int, updates_per_stage: int) -> float:
        """Coefficient for this update. ``update`` is the count already finished."""
        stage = update // updates_per_stage
        return self.entropy_coef[min(stage, len(self.entropy_coef) - 1)]


@dataclass
class LeagueConfig:
    run_dir: str = "runs/league"
    workers: int = 4
    torch_threads: int = 4
    seed: int = 42
    # Generations
    max_generations: int = 10
    stop_after_no_improvement: int = 2
    init_from: str = "scratch"  # "best", "previous" or "scratch"
    # PPO updates in each entropy stage. A generation runs every coefficient, then is evaluated.
    updates_per_gen: int = 300
    score_window: int = 1000
    # Opponents
    pfsp_power: float = 2.0
    pfsp_floor: float = 0.05
    train_opponent: dict = field(default_factory=lambda: {"kind": "ismcts", "iterations": 100})
    opponent_greedy: bool = False
    # Evaluation
    eval_opponent: dict = field(default_factory=lambda: {"kind": "ismcts", "iterations": 1000})
    eval_games: int = 500
    eval_greedy: bool = True
    # Games and saving
    max_actions: int = 300
    checkpoint_minutes: float = 15.0
    keep_checkpoints: int = 4
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> LeagueConfig:
        data = dict(data)
        for obsolete in ("min_updates", "max_updates", "patience", "improvement_margin"):
            data.pop(obsolete, None)
        ppo = PPOConfig(**data.pop("ppo", {}))
        return cls(**data, ppo=ppo)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ---------------------------------------------------------------- collection

_OPPONENTS: dict[str, object] = {}


def _opponent(spec: dict):
    key = json.dumps(spec, sort_keys=True)
    agent = _OPPONENTS.get(key)
    if agent is None:
        agent = _OPPONENTS[key] = make_agent(spec)
    return agent


def _sample(probs: np.ndarray, u: float) -> int:
    cumulative = np.cumsum(probs)
    index = int(np.searchsorted(cumulative, u * cumulative[-1], side="right"))
    return min(index, len(probs) - 1)


def collect(job: dict) -> dict:
    """Play ``job["games"]`` games of the learner against one opponent.

    Returns the learner's decisions (only those with more than one legal
    move) with advantages and returns, plus each game's score.
    """
    if "weights" in job:
        net = NumpyNet(job["weights"])
    else:
        net = load_numpy_net(job["weights_path"])
    opponent = _opponent(job["opponent"])
    rng = random.Random(job["seed"])
    lam = job["gae_lambda"]
    gamma = job["gamma"]
    max_actions = job["max_actions"]
    obs_rows, masks, actions, logps, advs, rets = [], [], [], [], [], []
    scores, lengths = [], []
    deal_seed = starter = 0
    for index in range(job["games"]):
        if index % 2 == 0:
            deal_seed = rng.randrange(2**62)
            starter = rng.randrange(2)
        me = index % 2
        game = FastGame(deal_seed, starter, max_actions=max_actions, record_events=True)
        trackers = (Tracker(0, game.discard[0]), Tracker(1, game.discard[0]))
        opponent.new_game(rng.randrange(2**48))
        values = []
        start = len(obs_rows)
        while not game.over:
            seat = game.current
            state = PublicState.from_fast(game, trackers[seat])
            if seat != me:
                game.step(opponent.act(state))
                continue
            legal = state.legal
            if len(legal) == 1:
                game.step(legal[0])
                continue
            x = encode(state)
            logits, value = net.forward(x)
            chosen = logits[legal].astype(np.float64)
            chosen -= chosen.max()
            probs = np.exp(chosen)
            probs /= probs.sum()
            pick = _sample(probs, rng.random())
            mask = np.zeros(NUM_ACTIONS, dtype=bool)
            mask[legal] = True
            obs_rows.append(x)
            masks.append(mask)
            actions.append(legal[pick])
            logps.append(float(np.log(probs[pick] + 1e-12)))
            values.append(float(value))
            game.step(legal[pick])
        reward = game.result_for(me)
        scores.append((reward + 1.0) / 2.0)
        lengths.append(game.n_actions)
        # GAE with a single reward at the end.
        steps = len(values)
        gae = 0.0
        game_adv = [0.0] * steps
        for t in range(steps - 1, -1, -1):
            next_value = values[t + 1] if t + 1 < steps else 0.0
            r = reward if t == steps - 1 else 0.0
            delta = r + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            game_adv[t] = gae
        advs.extend(game_adv)
        rets.extend(a + v for a, v in zip(game_adv, values))
        assert len(obs_rows) - start == steps
    count = len(obs_rows)
    return {
        "opponent": job["opponent_name"],
        "obs": np.stack(obs_rows) if count else np.zeros((0, INPUT_SIZE), np.float32),
        "mask": np.stack(masks) if count else np.zeros((0, NUM_ACTIONS), bool),
        "actions": np.array(actions, dtype=np.int64),
        "logp": np.array(logps, dtype=np.float32),
        "adv": np.array(advs, dtype=np.float32),
        "ret": np.array(rets, dtype=np.float32),
        "scores": scores,
        "lengths": lengths,
    }


# ---------------------------------------------------------------- update


def ppo_update(model, optimizer, batches: list[dict], cfg: PPOConfig, rng: np.random.Generator,
               entropy_coef: float) -> dict:
    import torch
    import torch.nn.functional as F

    obs = torch.from_numpy(np.concatenate([b["obs"] for b in batches]))
    mask = torch.from_numpy(np.concatenate([b["mask"] for b in batches]))
    actions = torch.from_numpy(np.concatenate([b["actions"] for b in batches]))
    old_logp = torch.from_numpy(np.concatenate([b["logp"] for b in batches]))
    adv = torch.from_numpy(np.concatenate([b["adv"] for b in batches]))
    ret = torch.from_numpy(np.concatenate([b["ret"] for b in batches]))
    n = obs.shape[0]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
    steps = 0
    stop = False
    model.train()
    for _ in range(cfg.epochs):
        order = torch.from_numpy(rng.permutation(n))
        for start in range(0, n, cfg.minibatch):
            idx = order[start : start + cfg.minibatch]
            logits, value = model(obs[idx], mask[idx])
            log_probs = F.log_softmax(logits, dim=-1)
            logp = log_probs.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
            ratio = torch.exp(logp - old_logp[idx])
            a = adv[idx]
            policy_loss = -torch.min(ratio * a, torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
            value_loss = F.mse_loss(value, ret[idx])
            probs = log_probs.exp()
            entropy = -(probs * log_probs.masked_fill(~mask[idx], 0.0)).sum(-1).mean()
            loss = policy_loss + cfg.value_coef * value_loss - entropy_coef * entropy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            with torch.no_grad():
                approx_kl = (old_logp[idx] - logp).mean().item()
                clip_frac = ((ratio - 1.0).abs() > cfg.clip).float().mean().item()
            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.item()
            stats["approx_kl"] += approx_kl
            stats["clip_frac"] += clip_frac
            steps += 1
            if cfg.target_kl and approx_kl > 1.5 * cfg.target_kl:
                stop = True
                break
        if stop:
            break
    for key in stats:
        stats[key] /= max(steps, 1)
    stats["samples"] = n
    stats["minibatches"] = steps
    return stats
