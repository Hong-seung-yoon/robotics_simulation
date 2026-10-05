from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def terrain_height_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terrain height below each ray relative to the robot root, minus a nominal standing height.

    Rays that miss the terrain mesh (outside the generated grid) are reported as flat ground.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    root_z = asset.data.root_pos_w[:, 2].unsqueeze(1)
    hits_z = sensor.data.ray_hits_w[..., 2]
    heights = root_z - hits_z - offset
    return torch.where(torch.isfinite(heights), heights, torch.zeros_like(heights))
