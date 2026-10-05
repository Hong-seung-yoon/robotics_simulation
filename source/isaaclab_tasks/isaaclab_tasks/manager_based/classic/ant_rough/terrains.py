"""Terrain configurations for Ant rough-terrain training and unseen-terrain evaluation."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

ANT_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=20,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.15, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.02, 0.1),
            step_width=0.35,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.02, 0.1),
            step_width=0.35,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.02, 0.08), platform_width=2.0
        ),
    },
)

ANT_UNSEEN_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=12,
    num_cols=5,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2, amplitude_range=(0.03, 0.12), num_waves=4, border_width=0.25
        ),
        "discrete_obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.2,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.3, 0.8),
            obstacle_height_range=(0.02, 0.08),
            num_obstacles=40,
            platform_width=1.5,
            border_width=0.25,
        ),
        "repeated_boxes": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=0.2,
            platform_width=1.5,
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=20, height=0.03, size=(0.3, 0.3)
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=50, height=0.08, size=(0.5, 0.5)
            ),
        ),
        "rough_hard": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.04, 0.10), noise_step=0.02, border_width=0.25
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.2,
            stone_height_max=0.03,
            stone_width_range=(0.5, 1.0),
            stone_distance_range=(0.05, 0.15),
            holes_depth=-0.3,
            platform_width=1.5,
            border_width=0.25,
        ),
    },
)

ANT_ROUGH_MILD_TERRAINS_CFG = ANT_ROUGH_TERRAINS_CFG.replace(
    num_cols=20,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.4),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.01, 0.05), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.25), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.25), platform_width=2.0, border_width=0.25
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.02, 0.08),
            step_width=0.35,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.05,
            step_height_range=(0.02, 0.08),
            step_width=0.35,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.02, 0.06), platform_width=2.0
        ),
    },
)

ANT_ALL_TERRAINS_CFG = ANT_ROUGH_TERRAINS_CFG.replace(
    num_cols=12,
    sub_terrains={
        **{k: v.replace(proportion=0.15 if k == "flat" else 0.0775) for k, v in ANT_ROUGH_TERRAINS_CFG.sub_terrains.items()},
        **{k: v.replace(proportion=0.0775) for k, v in ANT_UNSEEN_TERRAINS_CFG.sub_terrains.items()},
    },
)

ANT_UNSEEN2_TERRAINS_CFG = ANT_UNSEEN_TERRAINS_CFG.replace(
    sub_terrains={
        "repeated_cylinders": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            proportion=0.2,
            platform_width=1.5,
            object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=20, height=0.03, radius=0.15
            ),
            object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=50, height=0.08, radius=0.3
            ),
        ),
        "repeated_pyramids": terrain_gen.MeshRepeatedPyramidsTerrainCfg(
            proportion=0.2,
            platform_width=1.5,
            object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=20, height=0.04, radius=0.2
            ),
            object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                num_objects=50, height=0.1, radius=0.4
            ),
        ),
        "hf_stairs": terrain_gen.HfPyramidStairsTerrainCfg(
            proportion=0.2, step_height_range=(0.02, 0.08), step_width=0.4, platform_width=2.0, border_width=0.25
        ),
        "star": terrain_gen.MeshStarTerrainCfg(
            proportion=0.2, num_bars=6, bar_width_range=(0.3, 0.6), bar_height_range=(0.02, 0.08), platform_width=1.5
        ),
        "box_step": terrain_gen.MeshBoxTerrainCfg(
            proportion=0.2, box_height_range=(0.03, 0.1), platform_width=2.0, double_box=True
        ),
    },
)

ANT_GRID_TERRAINS_CFG = ANT_UNSEEN_TERRAINS_CFG.replace(
    num_rows=12,
    num_cols=4,
    sub_terrains={
        "grid_boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=1.0, grid_width=0.42, grid_height_range=(0.03, 0.15), platform_width=1.0
        ),
    },
)

ANT_ALL_GRIDHEAVY_TERRAINS_CFG = ANT_ALL_TERRAINS_CFG.replace(
    sub_terrains={
        **{k: v.replace(proportion=0.12 if k == "flat" else 0.058) for k, v in ANT_ALL_TERRAINS_CFG.sub_terrains.items()},
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.3, grid_width=0.42, grid_height_range=(0.02, 0.15), platform_width=1.5
        ),
    },
)

ANT_DIVERSE_TERRAINS_CFG = ANT_ALL_TERRAINS_CFG.replace(
    num_cols=16,
    sub_terrains={
        **{k: v.replace(proportion=0.10 if k == "flat" else 0.05) for k, v in ANT_ALL_TERRAINS_CFG.sub_terrains.items()},
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.42, grid_height_range=(0.02, 0.16), platform_width=1.5
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.08, slope_range=(0.05, 0.45), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.08, slope_range=(0.05, 0.45), platform_width=2.0, border_width=0.25
        ),
        "pillars": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            proportion=0.1,
            platform_width=1.5,
            object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=8, height=0.6, radius=0.12
            ),
            object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=24, height=0.6, radius=0.2
            ),
        ),
    },
)

ANT_UNSEEN3_HARD_TERRAINS_CFG = ANT_UNSEEN_TERRAINS_CFG.replace(
    num_cols=6,
    sub_terrains={
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=1 / 6, rail_thickness_range=(0.08, 0.15), rail_height_range=(0.04, 0.12), platform_width=1.5
        ),
        "star": terrain_gen.MeshStarTerrainCfg(
            proportion=1 / 6, num_bars=8, bar_width_range=(0.3, 0.6), bar_height_range=(0.04, 0.12), platform_width=1.5
        ),
        "box_step": terrain_gen.MeshBoxTerrainCfg(
            proportion=1 / 6, box_height_range=(0.05, 0.2), platform_width=1.5, double_box=True
        ),
        "hf_stairs_inv": terrain_gen.HfInvertedPyramidStairsTerrainCfg(
            proportion=1 / 6, step_height_range=(0.03, 0.1), step_width=0.4, platform_width=1.5, border_width=0.25
        ),
        "rough_xhard": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1 / 6, noise_range=(0.06, 0.14), noise_step=0.02, border_width=0.25
        ),
        "obstacles_tall": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=1 / 6,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.3, 0.8),
            obstacle_height_range=(0.06, 0.16),
            num_obstacles=40,
            platform_width=1.5,
            border_width=0.25,
        ),
    },
)

ANT_FINAL_TERRAINS_CFG = ANT_DIVERSE_TERRAINS_CFG.replace(
    sub_terrains={
        **{k: v.replace(proportion=0.10 if k == "flat" else 0.045) for k, v in ANT_DIVERSE_TERRAINS_CFG.sub_terrains.items()},
        "boxes": ANT_DIVERSE_TERRAINS_CFG.sub_terrains["boxes"].replace(proportion=0.18),
        "hf_pyramid_slope": ANT_DIVERSE_TERRAINS_CFG.sub_terrains["hf_pyramid_slope"].replace(proportion=0.08),
        "hf_pyramid_slope_inv": ANT_DIVERSE_TERRAINS_CFG.sub_terrains["hf_pyramid_slope_inv"].replace(proportion=0.08),
        "pillars": ANT_DIVERSE_TERRAINS_CFG.sub_terrains["pillars"].replace(proportion=0.15),
    },
)

ANT_UNSEEN4_OBSTACLES_TERRAINS_CFG = ANT_UNSEEN_TERRAINS_CFG.replace(
    num_cols=4,
    sub_terrains={
        "tall_boxes": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=0.25,
            platform_width=1.5,
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(num_objects=6, height=0.6, size=(0.3, 0.3)),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(num_objects=18, height=0.6, size=(0.5, 0.5)),
        ),
        "tall_cones": terrain_gen.MeshRepeatedPyramidsTerrainCfg(
            proportion=0.25,
            platform_width=1.5,
            object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(num_objects=6, height=0.6, radius=0.2),
            object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(num_objects=18, height=0.6, radius=0.35),
        ),
        "wave_slope": terrain_gen.HfWaveTerrainCfg(proportion=0.25, amplitude_range=(0.05, 0.2), num_waves=3, border_width=0.25),
        "steep_slope": terrain_gen.HfPyramidSlopedTerrainCfg(proportion=0.25, slope_range=(0.3, 0.55), platform_width=2.0, border_width=0.25),
    },
)
