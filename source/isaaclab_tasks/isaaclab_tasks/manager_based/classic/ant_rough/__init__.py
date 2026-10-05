"""Ant locomotion on rough terrain and unseen-terrain evaluation environments."""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Ant-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
    },
)

for _suffix, _cfg in [
    ("Unseen", "AntUnseenEnvCfg"),
    ("Unseen-Slippery", "AntUnseenSlipperyEnvCfg"),
    ("Unseen-Sticky", "AntUnseenStickyEnvCfg"),
    ("Flat-Slippery", "AntFlatSlipperyEnvCfg"),
    ("Flat-Sticky", "AntFlatStickyEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
        },
    )

for _suffix, _cfg in [
    ("Rough-TerrainOnly", "AntRoughTerrainOnlyEnvCfg"),
    ("Flat-PhysicsOnly", "AntFlatPhysicsOnlyEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntAblationPPORunnerCfg",
        },
    )

gym.register(
    id="Isaac-Ant-Rough-Mild-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRoughMildEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
    },
)

for _suffix, _cfg in [
    ("Rough-All", "AntRoughAllEnvCfg"),
    ("Rough-All-Stable", "AntRoughAllStableEnvCfg"),
    ("Unseen2", "AntUnseen2EnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
        },
    )

for _suffix, _cfg in [
    ("Rough-All-Scan", "AntRoughAllScanEnvCfg"),
    ("Unseen-Scan", "AntUnseenScanEnvCfg"),
    ("Unseen2-Scan", "AntUnseen2ScanEnvCfg"),
    ("Flat-Scan", "AntFlatScanEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
        },
    )

for _suffix, _cfg in [
    ("Flat-Scan-Slippery", "AntFlatScanSlipperyEnvCfg"),
    ("Flat-Scan-Sticky", "AntFlatScanStickyEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
        },
    )

for _suffix, _cfg in [
    ("Rough-All-Death", "AntRoughAllDeathEnvCfg"),
    ("Rough-All-Scan-Death", "AntRoughAllScanDeathEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
        },
    )

for _suffix, _cfg in [
    ("Grid", "AntGridEnvCfg"),
    ("Grid-Slippery", "AntGridSlipperyEnvCfg"),
    ("Grid-Scan", "AntGridScanEnvCfg"),
    ("Grid-Scan-Slippery", "AntGridScanSlipperyEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
        },
    )

for _suffix, _cfg in [
    ("Rough-All-Grid", "AntRoughAllGridEnvCfg"),
    ("Rough-All-Grid-Scan", "AntRoughAllGridScanEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
        },
    )

for _suffix, _cfg in [
    ("Diverse", "AntDiverseEnvCfg"),
    ("Diverse-Scan", "AntDiverseScanEnvCfg"),
    ("Diverse-Scan-Pd", "AntDiverseScanPdEnvCfg"),
    ("Unseen3-Hard", "AntUnseen3HardEnvCfg"),
    ("Unseen3-Hard-Scan", "AntUnseen3HardScanEnvCfg"),
    ("Unseen3-Hard-Scan-Pd", "AntUnseen3HardScanPdEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
        },
    )

for _suffix, _cfg in [
    ("Flat-Scan-Pd", "AntFlatScanPdEnvCfg"),
    ("Grid-Scan-Pd", "AntGridScanPdEnvCfg"),
    ("Grid-Scan-Slippery-Pd", "AntGridScanSlipperyPdEnvCfg"),
    ("Unseen2-Scan-Pd", "AntUnseen2ScanPdEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
        },
    )

gym.register(
    id="Isaac-Ant-Rough-All-Scan-Explore-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRoughAllScanEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughExplorePPORunnerCfg",
    },
)

for _suffix, _cfg in [
    ("Rough-All-Scan-Hist", "AntRoughAllScanHistEnvCfg"),
    ("Flat-Scan-Hist", "AntFlatScanHistEnvCfg"),
    ("Grid-Scan-Hist", "AntGridScanHistEnvCfg"),
    ("Grid-Scan-Slippery-Hist", "AntGridScanSlipperyHistEnvCfg"),
    ("Unseen2-Scan-Hist", "AntUnseen2ScanHistEnvCfg"),
    ("Unseen3-Hard-Scan-Hist", "AntUnseen3HardScanHistEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
        },
    )

gym.register(
    id="Isaac-Ant-Rough-All-Scan-Hist-Explore-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRoughAllScanHistEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughExplorePPORunnerCfg",
    },
)

for _suffix, _cfg in [
    ("Final-Eval", "AntFinalEvalEnvCfg"),
    ("Final-Flat", "AntFinalFlatEnvCfg"),
    ("Final-Obstacles", "AntFinalObstaclesEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
        },
    )

gym.register(
    id="Isaac-Ant-Rough-All-Scan-Hist-Death-Explore-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRoughAllScanHistDeathEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughExplorePPORunnerCfg",
    },
)

for _suffix, _cfg in [
    ("Rough-All-Scan-Hist-Smooth-Explore", "AntRoughAllScanHistSmoothEnvCfg"),
    ("Rough-All-Scan-Hist-PdSoft-Explore", "AntRoughAllScanHistPdSoftEnvCfg"),
    ("Unseen3-Hard-Scan-Hist-PdSoft", "AntUnseen3HardScanHistPdSoftEnvCfg"),
    ("Flat-Scan-Hist-PdSoft", "AntFlatScanHistPdSoftEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughExplorePPORunnerCfg",
        },
    )

for _suffix, _cfg in [
    ("Rough-All-Scan-Hist-Grounded-Explore", "AntRoughAllScanHistGroundedEnvCfg"),
    ("Rough-All-Scan-Hist-Grounded-LowTorque-Explore", "AntRoughAllScanHistGroundedLowTorqueEnvCfg"),
    ("Flat-Scan-Hist-LowTorque", "AntFlatScanHistLowTorqueEnvCfg"),
    ("Grid-Scan-Hist-LowTorque", "AntGridScanHistLowTorqueEnvCfg"),
    ("Grid-Scan-Slippery-Hist-LowTorque", "AntGridScanSlipperyHistLowTorqueEnvCfg"),
    ("Unseen2-Scan-Hist-LowTorque", "AntUnseen2ScanHistLowTorqueEnvCfg"),
    ("Unseen3-Hard-Scan-Hist-LowTorque", "AntUnseen3HardScanHistLowTorqueEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughExplorePPORunnerCfg",
        },
    )

for _suffix, _cfg in [
    ("Stress-Friction01", "AntStressFriction01EnvCfg"),
    ("Stress-Friction20", "AntStressFriction20EnvCfg"),
    ("Stress-Bouncy", "AntStressBouncyEnvCfg"),
    ("Stress-Yaw", "AntStressYawEnvCfg"),
    ("Stress-Friction01-LowTorque", "AntStressFriction01LowTorqueEnvCfg"),
    ("Stress-Friction20-LowTorque", "AntStressFriction20LowTorqueEnvCfg"),
    ("Stress-Bouncy-LowTorque", "AntStressBouncyLowTorqueEnvCfg"),
    ("Stress-Yaw-LowTorque", "AntStressYawLowTorqueEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughExplorePPORunnerCfg",
        },
    )

# Baseline PPO settings (no entropy change); iteration count is passed on the command line.
for _suffix, _cfg in [
    ("Rough-All-Scan-Hist-Smooth", "AntRoughAllScanHistSmoothEnvCfg"),
    ("Rough-All-Scan-Hist-Grounded", "AntRoughAllScanHistGroundedEnvCfg"),
    ("Rough-All-Scan-Hist-Grounded-LowTorque", "AntRoughAllScanHistGroundedLowTorqueEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
        },
    )

for _suffix, _cfg in [
    ("Final-Train", "AntFinalTrainEnvCfg"),
    ("Final-Train-Smooth", "AntFinalTrainSmoothEnvCfg"),
    ("Final-Train-Grounded", "AntFinalTrainGroundedEnvCfg"),
    ("Final-Train-Grounded-LowTorque", "AntFinalTrainGroundedLowTorqueEnvCfg"),
    ("Unseen4-Obstacles-Scan-Hist", "AntUnseen4ScanHistEnvCfg"),
    ("Unseen4-Obstacles-Scan-Hist-LowTorque", "AntUnseen4ScanHistLowTorqueEnvCfg"),
]:
    gym.register(
        id=f"Isaac-Ant-{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:{_cfg}",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
        },
    )

gym.register(
    id="Isaac-Ant-Final-Train-Grounded-Lite-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntFinalTrainGroundedLiteEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
    },
)
