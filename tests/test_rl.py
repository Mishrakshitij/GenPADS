"""Numerical policy checks and symbolic task/reward integration tests."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from genpads.domains import DOMAINS
from genpads.environment import Environment
from genpads.policy import ReinforcePolicy, Transition, discounted_returns, masked_probabilities
from genpads.rewards import baseline_reward, prrp_reward
from genpads.simulation import simulate, slot_filling_action, train_policy


class RewardTests(unittest.TestCase):
    def test_terminal_and_prrp_branches(self):
        self.assertEqual(baseline_reward(success=True), 20)
        self.assertEqual(baseline_reward(failure=True), -10)
        self.assertEqual(baseline_reward(), -1)
        for label in (0, 1):
            self.assertEqual(prrp_reward(agent_label=label, noisy=True), -2)
            self.assertEqual(prrp_reward(agent_label=label, noisy=False), -1)
        for label in (2, 3):
            self.assertEqual(prrp_reward(agent_label=label, noisy=True), -.5)
            self.assertEqual(prrp_reward(agent_label=label, noisy=False), -.5)
        self.assertEqual(prrp_reward(repeated=True, agent_label=3), -2.5)
        self.assertEqual(prrp_reward(success=True, repeated=True), 20)
        self.assertEqual(prrp_reward(failure=True, repeated=True), -10)
        with self.assertRaises(ValueError):
            prrp_reward(agent_label=4)

    def test_discounted_returns(self):
        np.testing.assert_allclose(discounted_returns([-1, -1, 20], .9), [14.3, 17, 20])

    def test_mask_and_exploration_probabilities(self):
        actual = masked_probabilities([np.log(3), 10_000, 0], [True, False, True], exploration=.2)
        np.testing.assert_allclose(actual, [.7, 0, .3])
        with self.assertRaises(ValueError):
            masked_probabilities([1, 2], [False, False])


class EnvironmentTests(unittest.TestCase):
    def test_paper_domain_counts_and_complete_tasks(self):
        expected = {"flights": (5, 12), "food-ordering": (5, 12), "hotels": (5, 12),
                    "movies": (4, 10), "music": (4, 10), "restaurant-search": (4, 10),
                    "sports": (5, 12)}
        for name, counts in expected.items():
            domain = DOMAINS[name]
            self.assertEqual((len(domain.slots), domain.num_actions), counts)
            env = Environment(name, seed=3)
            state = env.reset()
            self.assertEqual(state.shape, (domain.state_dim,))
            self.assertFalse(env.action_mask()[-1])
            with self.assertRaises(ValueError):
                env.step(domain.num_actions - 1)
            while not env.done:
                _, reward, _, info = env.step(slot_filling_action(env))
            self.assertTrue(env.success)
            self.assertEqual(reward, 20)
            self.assertEqual(len(info["known_slots"]), len(domain.slots))
            self.assertFalse(env.action_mask().any())
            with self.assertRaises(RuntimeError):
                env.step(0)

    def test_generated_text_controls_scoring_and_user_noise(self):
        seen = []
        def classifier(text):
            seen.append(text)
            return 3 if text == "A changed polite response." else 0
        env = Environment("flights", seed=0, generator=lambda _: "A changed polite response.",
                          classifier=classifier)
        informative = 0
        for _ in range(4000):
            env.reset()
            _, reward, _, info = env.step(0)  # The underlying template has label 0.
            informative += not info["noisy"]
            self.assertEqual(info["agent_label"], 3)
            self.assertEqual(reward, -.5)
            self.assertIn(info["user_utterance"], seen[-2:])
        self.assertAlmostEqual(informative / 4000, .9, delta=.02)
        self.assertIn("A changed polite response.", seen)
        with self.assertRaises(ValueError):
            Environment("flights", generator=lambda text: text)

    def test_noise_probability_for_impolite_requests(self):
        env = Environment("flights", seed=0)
        informative = 0
        for _ in range(4000):
            env.reset()
            informative += not env.step(0)[3]["noisy"]
        self.assertAlmostEqual(informative / 4000, .8, delta=.02)

    def test_repetition_and_timeout(self):
        env = Environment("flights", seed=0, max_turns=3)
        env.reset()
        self.assertFalse(env.step(1)[3]["noisy"])
        _, reward, _, info = env.step(0)
        self.assertTrue(info["repeated"])
        self.assertEqual(reward, -2.5)
        _, reward, done, info = env.step(0)
        self.assertTrue(done)
        self.assertTrue(info["failure"])
        self.assertEqual(reward, -10)

    def test_simulation_is_seeded(self):
        first = simulate("sports", episodes=10, seed=42, record_dialogues=True)
        self.assertEqual(first, simulate("sports", episodes=10, seed=42, record_dialogues=True))
        self.assertEqual(first["scoring_mode"], "template_metadata")

    def test_evaluation_restores_generator_sampling_stream(self):
        class Generator:
            def __init__(self):
                self.reseed(12)

            def reseed(self, seed):
                self.rng = np.random.default_rng(seed)

            def get_rng_state(self):
                return deepcopy(self.rng.bit_generator.state)

            def set_rng_state(self, state):
                self.rng.bit_generator.state = state

            def __call__(self, text):
                return f"{text} Thank you {self.rng.integers(100000)}."

        generator = Generator()
        before = generator.get_rng_state()
        first = simulate("music", episodes=2, seed=4, generator=generator,
                         classifier=lambda _: 3, record_dialogues=True)
        self.assertEqual(before, generator.get_rng_state())
        self.assertEqual(first, simulate("music", episodes=2, seed=4, generator=generator,
                                         classifier=lambda _: 3, record_dialogues=True))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is an optional training dependency")
class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        # Tiny recurrent examples are much faster without large thread pools.
        torch.set_num_threads(1)

    def test_exploration_is_in_loss_and_masked_gradient(self):
        policy = ReinforcePolicy(2, 3, seed=7)
        torch = policy.torch
        with torch.no_grad():
            for parameter in policy.model.parameters():
                parameter.zero_()
            policy.model["head"].bias.copy_(torch.tensor([np.log(3), 9., 0.]))
        step = Transition.snapshot([0, 0], 0, 2, [True, False, True], .2)
        loss = policy.update([step])
        self.assertAlmostEqual(loss, -2 * np.log(.7), places=5)
        self.assertEqual(float(policy.model["head"].bias.grad[1]), 0)
        self.assertLess(float(policy.model["head"].bias.grad[0]), 0)
        # All-uniform exploration is independent of the policy: zero gradient.
        policy.update([Transition.snapshot([0, 0], 0, 2, [True, False, True], 1)])
        for parameter in policy.model.parameters():
            self.assertEqual(float(parameter.grad.abs().sum()), 0)

    def test_training_checkpoint_and_frozen_evaluation(self):
        domain = DOMAINS["movies"]
        policy = ReinforcePolicy(domain.state_dim, domain.num_actions, seed=8)
        original = {name: value.clone() for name, value in policy.model.state_dict().items()}
        with tempfile.TemporaryDirectory() as directory:
            result = train_policy("movies", episodes=3, evaluate_every=2, evaluation_episodes=2,
                                  seed=8, policy=policy, output=directory)
            self.assertEqual(policy.updates, 3)
            self.assertEqual([row["episode"] for row in result["history"]], [2, 3])
            self.assertTrue(Path(result["checkpoint"]).is_file())
            restored = ReinforcePolicy.load(result["checkpoint"])
            self.assertEqual(restored.updates, 3)
            self.assertEqual(restored.domain_name, "movies")
            with self.assertRaises(ValueError):
                simulate("music", episodes=1, policy=restored)
            self.assertTrue(any(not policy.torch.equal(value, original[name])
                                for name, value in policy.model.state_dict().items()))
            for name, value in policy.model.state_dict().items():
                self.assertTrue(policy.torch.equal(value, restored.model.state_dict()[name]))
            rng_before = deepcopy(restored.rng.bit_generator.state)
            weights_before = {name: value.clone() for name, value in restored.model.state_dict().items()}
            simulate("movies", episodes=2, seed=11, policy=restored)
            self.assertEqual(restored.rng.bit_generator.state, rng_before)
            for name, value in restored.model.state_dict().items():
                self.assertTrue(policy.torch.equal(value, weights_before[name]))


if __name__ == "__main__":
    unittest.main()
