from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def root_height_below_minimum_relative(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the root height above the terrain is too low.

    The terrain height is measured by a downward ray caster. Where no ray hits the terrain mesh
    (outside the generated grid), the world-frame ground height of 0.0 is used, which matches the
    baseline flat-terrain check.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    hits_z = sensor.data.ray_hits_w[..., 2]
    hits_z = torch.where(torch.isfinite(hits_z), hits_z, torch.nan)
    ground_z = torch.nanmean(hits_z, dim=1)
    root_z = asset.data.root_pos_w[:, 2]
    ground_z = torch.nan_to_num(ground_z, nan=0.0)
    return (root_z - ground_z) < minimum_height
