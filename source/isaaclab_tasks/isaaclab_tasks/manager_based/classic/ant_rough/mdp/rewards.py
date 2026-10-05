from __future__ import annotations

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def fall_termination(env: ManagerBasedRLEnv) -> torch.Tensor:
    """One-off signal on the step where an episode ends by a non-timeout termination.

    Reads the per-step ``terminated`` buffer. The per-term buffer used by ``is_terminated_term``
    is not cleared every step in this IsaacLab version, so it would fire for the whole next episode.
    """
    return env.termination_manager.terminated.float()
