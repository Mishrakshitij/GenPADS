"""Baseline and PRRP immediate rewards from Algorithms 2 and 3."""

from numbers import Integral


def validate_label(label: int) -> int:
    """Accept canonical classifier labels only; archive conversion belongs to data IO."""
    if isinstance(label, bool) or not isinstance(label, Integral) or not 0 <= label <= 3:
        raise ValueError("Politeness labels must be canonical integers 0, 1, 2, or 3.")
    return int(label)


def baseline_reward(*, success=False, failure=False) -> float:
    if success and failure:
        raise ValueError("A turn cannot be both successful and failed.")
    return 20.0 if success else -10.0 if failure else -1.0


def prrp_reward(*, success=False, failure=False, repeated=False,
                noisy=False, agent_label=0, slot_request=True) -> float:
    """Penalize redundant requests even if the user supplies the same slot again.

    Terminal rewards have priority. A retry after noise is not a repetition;
    the environment marks requests for already supplied slots as repeated.
    """
    agent_label = validate_label(agent_label)
    if success or failure:
        return baseline_reward(success=success, failure=failure)
    if repeated:
        return -2.5
    if slot_request:
        if agent_label >= 2:
            return -0.5
        return -2.0 if noisy else -1.0
    return -1.0
