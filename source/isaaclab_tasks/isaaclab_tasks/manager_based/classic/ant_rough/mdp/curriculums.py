from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_levels_progress(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    up_distance: float = 6.0,
    down_distance: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terrain curriculum driven by forward progress along the +x target direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    distance = asset.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    move_up = distance > up_distance
    move_down = distance < down_distance
    move_down *= ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
