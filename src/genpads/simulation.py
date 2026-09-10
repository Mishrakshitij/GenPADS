"""Runnable retrieval baseline, frozen-policy evaluation, and RL training."""

from numbers import Integral

import numpy as np

from .domains import get_domain
from .environment import Environment
from .policy import ReinforcePolicy, Transition


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def _check_policy_domain(policy, domain):
    if (policy.state_dim != domain.state_dim or policy.num_actions != domain.num_actions
            or policy.domain_name not in (None, domain.name)):
        raise ValueError("Policy does not match the selected domain.")


def slot_filling_action(environment, *, polite=True):
    """Deterministic task sanity check, separate from the learned policy."""
    for index, slot in enumerate(environment.domain.slots):
        if slot not in environment.known_slots:
            return 2 * index + int(polite)
    return environment.domain.num_actions - (1 if polite else 2)


def _episode(environment, policy=None, *, exploration=0.0, greedy=False,
             polite=True, record_dialogue=False):
    state = environment.reset()
    recurrent = policy.initial_state() if policy is not None else None
    trajectory, dialogue = [], []
    total_reward = 0.0
    polite_turns = 0
    while not environment.done:
        mask = environment.action_mask()
        if policy is None:
            action = slot_filling_action(environment, polite=polite)
        else:
            action, recurrent = policy.act(state, recurrent, mask,
                                           exploration=exploration, greedy=greedy)
        following, reward, _, info = environment.step(action)
        trajectory.append(Transition.snapshot(state, action, reward, mask, exploration))
        total_reward += reward
        polite_turns += info["agent_label"] >= 2
        if record_dialogue:
            dialogue.append(dict(info, reward=reward))
        state = following
    result = {
        "success": environment.success, "turns": environment.turns,
        "reward": total_reward, "politeness": polite_turns / environment.turns,
    }
    if record_dialogue:
        result["dialogue"] = dialogue
    return result, trajectory


def simulate(domain, *, episodes=100, seed=0, policy=None, reward_mode="prrp",
             generator=None, classifier=None, max_turns=None, greedy=False,
             polite=True, record_dialogues=False):
    """Evaluate without updating weights or advancing the policy sampling RNG."""
    _positive_integer(episodes, "episodes")
    environment = Environment(domain, seed=seed, reward_mode=reward_mode,
                              generator=generator, classifier=classifier, max_turns=max_turns)
    if policy is not None:
        _check_policy_domain(policy, environment.domain)
    # Evaluation uses its own repeatable stream and cannot perturb training RNG.
    previous_rng = policy.rng if policy is not None else None
    previous_mode = policy.model.training if policy is not None else None
    generator_has_rng = generator is not None and all(callable(getattr(generator, name, None))
                        for name in ("get_rng_state", "set_rng_state", "reseed"))
    previous_generator_rng = generator.get_rng_state() if generator_has_rng else None
    try:
        if policy is not None:
            policy.rng = np.random.default_rng(seed + 1)
        if generator_has_rng:
            generator.reseed(seed + 2)
        results = [_episode(environment, policy, greedy=greedy, polite=polite,
                            record_dialogue=record_dialogues)[0] for _ in range(episodes)]
    finally:
        if generator_has_rng:
            generator.set_rng_state(previous_generator_rng)
        if policy is not None:
            policy.rng = previous_rng
            policy.model.train(previous_mode)
    report = {
        "domain": environment.domain.name, "episodes": int(episodes), "seed": int(seed),
        "reward_mode": reward_mode, "response_mode": environment.response_mode,
        "scoring_mode": environment.scoring_mode,
        "policy": "learned" if policy is not None else "slot_filling_sanity_check",
        "success_rate": float(np.mean([row["success"] for row in results])),
        "mean_turns": float(np.mean([row["turns"] for row in results])),
        "mean_reward": float(np.mean([row["reward"] for row in results])),
        "mean_politeness": float(np.mean([row["politeness"] for row in results])),
    }
    if record_dialogues:
        report["dialogues"] = results
    return report


def train_policy(domain, *, episodes=8000, evaluate_every=400,
                 evaluation_episodes=500, seed=0, reward_mode="prrp",
                 generator=None, classifier=None, max_turns=None, exploration=0.1,
                 learning_rate=1.0, discount=0.9, hidden_dim=32, device="cpu",
                 output=None, policy=None, progress=None):
    """Train after each dialogue and freeze weights for periodic evaluation.

    The paper does not specify epsilon, AdaDelta learning rate, or an exploration
    schedule. Here epsilon is a constant, explicitly configurable probability.
    """
    for value, name in ((episodes, "episodes"), (evaluate_every, "evaluate_every"),
                        (evaluation_episodes, "evaluation_episodes")):
        _positive_integer(value, name)
    if not 0 <= exploration <= 1:
        raise ValueError("exploration must be in [0, 1].")
    specification = get_domain(domain)
    environment = Environment(specification, seed=seed, reward_mode=reward_mode,
                              generator=generator, classifier=classifier, max_turns=max_turns)
    if policy is None:
        policy = ReinforcePolicy(specification.state_dim, specification.num_actions,
                                 hidden_dim=hidden_dim, learning_rate=learning_rate,
                                 discount=discount, seed=seed, device=device)
    _check_policy_domain(policy, specification)
    policy.domain_name = specification.name
    history = []
    losses = []
    for episode in range(1, episodes + 1):
        _, trajectory = _episode(environment, policy, exploration=exploration)
        losses.append(policy.update(trajectory))
        if episode % evaluate_every == 0 or episode == episodes:
            evaluation = simulate(specification, episodes=evaluation_episodes,
                                  seed=seed + 1_000_000, policy=policy,
                                  reward_mode=reward_mode, generator=generator,
                                  classifier=classifier, max_turns=max_turns)
            history.append({"episode": episode, "mean_loss": float(np.mean(losses)),
                            **evaluation})
            losses.clear()
            if progress is not None:
                progress(history[-1])
    checkpoint = policy.save(output) if output is not None else None
    return {"domain": specification.name, "episodes": int(episodes),
            "seed": int(seed), "exploration": float(exploration),
            "checkpoint": checkpoint, "history": history}
