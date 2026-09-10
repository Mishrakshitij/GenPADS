"""Seeded symbolic user simulation with optional learned generation and scoring.

No external model is required for the retrieval baseline. In that mode the
known template style supplies the label and is reported as template metadata,
not as a prediction by a learned politeness classifier.
"""

from collections.abc import Callable
from numbers import Integral

import numpy as np

from .domains import DomainSpec, get_domain
from .rewards import baseline_reward, prrp_reward, validate_label


class Environment:
    """A single-domain episode; slots are symbolic placeholders, not a live service.

    ``generator(template) -> str`` and ``classifier(text) -> int`` are injectable.
    The emitted response is classified before its label determines user noise
    and reward. Symbolic action intent is assumed preserved by the generator.
    """

    def __init__(self, domain: str | DomainSpec, *, seed=0, reward_mode="prrp",
                 generator: Callable[[str], str] | None = None,
                 classifier: Callable[[str], int] | None = None, max_turns=None):
        self.domain = get_domain(domain)
        if reward_mode not in {"baseline", "prrp"}:
            raise ValueError("reward_mode must be 'baseline' or 'prrp'.")
        if generator is not None and classifier is None:
            raise ValueError("Generated responses require a politeness classifier.")
        self.max_turns = self.domain.max_turns if max_turns is None else max_turns
        if isinstance(self.max_turns, bool) or not isinstance(self.max_turns, Integral) or self.max_turns < 1:
            raise ValueError("max_turns must be a positive integer.")
        self.rng = np.random.default_rng(seed)
        self.reward_mode = reward_mode
        self.generator = generator
        self.classifier = classifier
        self.response_mode = "generated" if generator is not None else "retrieval"
        self.scoring_mode = "classifier" if classifier is not None else "template_metadata"
        self.known_slots: set[str] = set()
        self.state = np.zeros(self.domain.state_dim, dtype=np.float32)
        self.done = True
        self.success = False
        self.turns = 0
        self.user_utterance = ""
        self.last_action = None

    def _label(self, text, fallback):
        return validate_label(self.classifier(text) if self.classifier is not None else fallback)

    def _state(self, user_label):
        self.state.fill(0)
        for index, slot in enumerate(self.domain.slots):
            self.state[index] = slot in self.known_slots
        offset = len(self.domain.slots)
        if self.last_action is not None:
            self.state[offset + self.last_action] = 1
        self.state[offset + self.domain.num_actions + user_label] = 1
        return self.state.copy()

    def reset(self):
        self.known_slots.clear()
        self.done = self.success = False
        self.turns = 0
        self.last_action = None
        self.user_utterance = f"Please help me with {self.domain.task.replace('_', ' ')}."
        return self._state(self._label(self.user_utterance, 2))

    def action_mask(self):
        # Repeated requests remain legal so PRRP can penalize them. Completion
        # requires all slots to be observed; no hidden goal information leaks.
        all_known = len(self.known_slots) == len(self.domain.slots)
        return np.array([
            not self.done and (not action.is_completion or all_known)
            for action in self.domain.actions
        ], dtype=bool)

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode has ended; call reset() before stepping.")
        if isinstance(action, bool) or not isinstance(action, Integral):
            raise ValueError("Action must be an integer index.")
        if not 0 <= action < self.domain.num_actions or not self.action_mask()[action]:
            raise ValueError("Action is outside the action space or prohibited by its mask.")
        action = int(action)
        specification = self.domain.actions[action]
        utterance = (self.generator(specification.template) if self.generator is not None
                     else specification.template)
        if not isinstance(utterance, str) or not utterance.strip():
            raise ValueError("The generator must return a nonempty string.")
        agent_label = self._label(utterance, specification.politeness)
        repeated = specification.slot in self.known_slots
        noisy = False
        supplied_slot = None
        success = specification.is_completion
        if success:
            user_utterance = "Thank you, that is what I wanted."
            user_label = self._label(user_utterance, 3)
        else:
            informative_probability = 0.9 if agent_label >= 2 else 0.8
            noisy = bool(self.rng.random() >= informative_probability)
            if noisy:
                user_utterance = "That is not what I asked."
                user_label = self._label(user_utterance, 0)
            else:
                supplied_slot = specification.slot
                user_utterance = f"The {specification.slot.replace('_', ' ')} is <{specification.slot}>."
                user_label = self._label(user_utterance, 2)
        # Commit the transition after generation and both classifier calls work.
        if supplied_slot is not None:
            self.known_slots.add(supplied_slot)
        self.turns += 1
        self.success = success
        self.done = success or self.turns >= self.max_turns
        failure = self.done and not success
        self.last_action = action
        self.user_utterance = user_utterance
        if self.reward_mode == "baseline":
            reward = baseline_reward(success=success, failure=failure)
        else:
            reward = prrp_reward(success=success, failure=failure, repeated=repeated,
                                 noisy=noisy, agent_label=agent_label,
                                 slot_request=not specification.is_completion)
        info = {
            "turn": self.turns, "action": specification.name,
            "template": specification.template, "agent_utterance": utterance,
            "user_utterance": user_utterance, "agent_label": agent_label,
            "user_label": user_label, "noisy": noisy, "repeated": repeated,
            "supplied_slot": supplied_slot, "success": success, "failure": failure,
            "known_slots": sorted(self.known_slots),
        }
        return self._state(user_label), float(reward), self.done, info
