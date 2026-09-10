"""Masked LSTM REINFORCE with lazy PyTorch loading and AdaDelta updates."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def discounted_returns(rewards, discount=0.9):
    if not 0 <= discount <= 1:
        raise ValueError("discount must be in [0, 1].")
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.ndim != 1 or not np.isfinite(rewards).all():
        raise ValueError("Rewards must be a finite vector.")
    result = np.zeros(len(rewards), dtype=np.float32)
    total = 0.0
    for index in reversed(range(len(rewards))):
        total = float(rewards[index]) + discount * total
        result[index] = total
    return result


def masked_probabilities(logits, mask, *, exploration=0.0):
    """Mix masked policy probabilities with uniform valid-action exploration."""
    logits = np.asarray(logits, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    if logits.ndim != 1 or logits.shape != mask.shape or not mask.any():
        raise ValueError("Logits and mask must be matching vectors with a valid action.")
    if not np.isfinite(logits[mask]).all():
        raise ValueError("Valid logits must be finite.")
    if not 0 <= exploration <= 1:
        raise ValueError("exploration must be in [0, 1].")
    weights = np.zeros_like(logits)
    weights[mask] = np.exp(logits[mask] - logits[mask].max())
    return (1 - exploration) * weights / weights.sum() + exploration * mask / mask.sum()


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    mask: np.ndarray
    exploration: float = 0.0

    @classmethod
    def snapshot(cls, state, action, reward, mask, exploration=0.0):
        state_copy = np.array(state, dtype=np.float32, copy=True)
        mask_copy = np.array(mask, dtype=bool, copy=True)
        state_copy.flags.writeable = False
        mask_copy.flags.writeable = False
        return cls(state_copy, int(action), float(reward), mask_copy, float(exploration))


class ReinforcePolicy:
    """One recurrent policy per domain, trained from complete episodes.

    Sampling uses an epsilon mixture of the policy and uniform legal actions.
    Updates differentiate the same mixture, including the exploration term.
    This stochastic policy-gradient implementation is not a pure argmax
    epsilon-greedy Q-learning policy.
    """

    def __init__(self, state_dim, num_actions, *, hidden_dim=32, learning_rate=1.0,
                 discount=0.9, max_gradient=5.0, seed=0, device="cpu"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Policy training requires PyTorch; install the training extra.") from exc
        if min(state_dim, num_actions, hidden_dim) < 1 or learning_rate <= 0 or max_gradient <= 0:
            raise ValueError("Dimensions, learning_rate, and max_gradient must be positive.")
        if not 0 <= discount <= 1:
            raise ValueError("discount must be in [0, 1].")
        self.torch = torch
        self.device = torch.device(device)
        self.state_dim, self.num_actions = int(state_dim), int(num_actions)
        self.hidden_dim, self.discount = int(hidden_dim), float(discount)
        self.max_gradient = float(max_gradient)
        self.learning_rate = float(learning_rate)
        self.seed = int(seed)
        self.domain_name = None
        self.rng = np.random.default_rng(seed)
        # Keep initialization deterministic without changing callers' CPU RNG.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            self.model = torch.nn.ModuleDict({
                "lstm": torch.nn.LSTM(self.state_dim, self.hidden_dim, batch_first=True),
                "head": torch.nn.Linear(self.hidden_dim, self.num_actions),
            }).to(self.device)
        self.optimizer = torch.optim.Adadelta(self.model.parameters(), lr=learning_rate)
        self.updates = 0

    def initial_state(self):
        shape = (1, 1, self.hidden_dim)
        return tuple(self.torch.zeros(shape, device=self.device) for _ in range(2))

    def _forward(self, states, recurrent=None):
        features, recurrent = self.model["lstm"](states, recurrent)
        return self.model["head"](features), recurrent

    def act(self, state, recurrent, mask, *, exploration=0.0, greedy=False):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.state_dim,) or not np.isfinite(state).all():
            raise ValueError("State must be a finite vector matching state_dim.")
        self.model.eval()
        with self.torch.no_grad():
            tensor = self.torch.as_tensor(state, device=self.device)[None, None, :]
            logits, following = self._forward(tensor, recurrent)
        probabilities = masked_probabilities(logits[0, 0].cpu().numpy(), mask,
                                            exploration=exploration)
        action = int(probabilities.argmax()) if greedy else int(self.rng.choice(self.num_actions, p=probabilities))
        return action, tuple(value.detach() for value in following)

    def update(self, trajectory):
        if not trajectory:
            raise ValueError("Cannot update with an empty trajectory.")
        for step in trajectory:
            if step.state.shape != (self.state_dim,) or step.mask.shape != (self.num_actions,):
                raise ValueError("Trajectory state/mask dimensions do not match the policy.")
            if not np.isfinite(step.state).all() or not 0 <= step.exploration <= 1:
                raise ValueError("Trajectory contains an invalid state or exploration value.")
            if not 0 <= step.action < self.num_actions or not step.mask[step.action]:
                raise ValueError("Trajectory contains a masked or invalid action.")
        torch = self.torch
        returns = discounted_returns([step.reward for step in trajectory], self.discount)
        states = torch.as_tensor(np.stack([step.state for step in trajectory]), device=self.device)[None]
        masks = torch.as_tensor(np.stack([step.mask for step in trajectory]), device=self.device)
        actions = torch.tensor([step.action for step in trajectory], device=self.device)
        exploration = torch.tensor([step.exploration for step in trajectory], device=self.device)[:, None]
        self.model.train()
        self.optimizer.zero_grad()
        logits, _ = self._forward(states)
        probabilities = logits[0].masked_fill(~masks, float("-inf")).softmax(dim=-1)
        uniform = masks.float() / masks.sum(dim=-1, keepdim=True)
        behavior = (1 - exploration) * probabilities + exploration * uniform
        selected = behavior.gather(1, actions[:, None]).squeeze(1)
        # Clamping protects logarithms under extreme finite logits.
        log_probability = selected.clamp_min(torch.finfo(selected.dtype).tiny).log()
        loss = -(log_probability * torch.as_tensor(returns, device=self.device)).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite policy loss.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_gradient,
                                       error_if_nonfinite=True)
        self.optimizer.step()
        self.updates += 1
        return float(loss.detach().cpu())

    def save(self, path):
        path = Path(path).expanduser()
        if path.suffix != ".pt":
            path = path / "policy.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.torch.save({
            "format_version": 1,
            "config": {"state_dim": self.state_dim, "num_actions": self.num_actions,
                       "hidden_dim": self.hidden_dim, "learning_rate": self.learning_rate,
                       "discount": self.discount, "max_gradient": self.max_gradient,
                       "seed": self.seed},
            "model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
            "updates": self.updates, "rng_state": self.rng.bit_generator.state,
            "domain_name": self.domain_name,
        }, path)
        return str(path)

    @classmethod
    def load(cls, path, *, device="cpu"):
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Loading a policy requires PyTorch.") from exc
        path = Path(path).expanduser()
        if path.is_dir():
            path = path / "policy.pt"
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        if checkpoint.get("format_version") != 1:
            raise ValueError("Unsupported policy checkpoint format.")
        policy = cls(**checkpoint["config"], device=device)
        policy.model.load_state_dict(checkpoint["model"])
        policy.optimizer.load_state_dict(checkpoint["optimizer"])
        policy.updates = int(checkpoint["updates"])
        policy.domain_name = checkpoint.get("domain_name")
        policy.rng.bit_generator.state = checkpoint["rng_state"]
        return policy
