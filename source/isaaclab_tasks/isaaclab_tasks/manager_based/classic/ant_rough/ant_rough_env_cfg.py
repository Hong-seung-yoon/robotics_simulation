"""Ant locomotion on procedurally generated rough terrain with domain randomization.

Observation, action and reward definitions are identical to ``Isaac-Ant-v0`` so that a policy
trained here can be evaluated with the original flat-terrain configuration and vice versa.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import (
    AntEnvCfg,
    EventCfg,
    MySceneCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)

from . import mdp
from .terrains import (
    ANT_ALL_GRIDHEAVY_TERRAINS_CFG,
    ANT_ALL_TERRAINS_CFG,
    ANT_DIVERSE_TERRAINS_CFG,
    ANT_FINAL_TERRAINS_CFG,
    ANT_UNSEEN4_OBSTACLES_TERRAINS_CFG,
    ANT_UNSEEN3_HARD_TERRAINS_CFG,
    ANT_GRID_TERRAINS_CFG,
    ANT_ROUGH_MILD_TERRAINS_CFG,
    ANT_ROUGH_TERRAINS_CFG,
    ANT_UNSEEN2_TERRAINS_CFG,
    ANT_UNSEEN_TERRAINS_CFG,
)


@configclass
class AntRoughSceneCfg(MySceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ANT_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=2,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="average",
            restitution_combine_mode="average",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 2.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.2, 0.2]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class RoughEventCfg(EventCfg):
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.5),
            "dynamic_friction_range": (0.3, 1.5),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )

    scale_torso_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
        },
    )

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={"velocity_range": {"x": (-0.7, 0.7), "y": (-0.7, 0.7)}},
    )

    def __post_init__(self):
        self.reset_base.params["pose_range"] = {"yaw": (-0.3, 0.3)}


@configclass
class RoughTerminationsCfg(TerminationsCfg):
    torso_height = DoneTerm(
        func=mdp.root_height_below_minimum_relative,
        params={"minimum_height": 0.31, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_progress)


@configclass
class AntRoughEnvCfg(AntEnvCfg):
    scene: AntRoughSceneCfg = AntRoughSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    events: RoughEventCfg = RoughEventCfg()
    terminations: RoughTerminationsCfg = RoughTerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt


@configclass
class AntRoughEnvCfg_PLAY(AntRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.terrain.terrain_generator.num_rows = 12
        self.scene.terrain.terrain_generator.num_cols = 10
        self.scene.terrain.max_init_terrain_level = 0
        self.curriculum.terrain_levels = None
        self.events.push_robot = None


@configclass
class AntUnseenEnvCfg(AntRoughEnvCfg):
    """Evaluation on terrain types never seen during training, without robot-side randomization.

    All robots start on the easiest row and walk along +x into progressively harder rows, so the
    episode return reflects how far the policy gets before falling.
    """

    terrain_friction: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.terrain.terrain_generator = ANT_UNSEEN_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.physics_material.static_friction = self.terrain_friction
        self.scene.terrain.physics_material.dynamic_friction = self.terrain_friction
        self.curriculum.terrain_levels = None
        self.events.physics_material = None
        self.events.scale_torso_mass = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {}


@configclass
class AntUnseenSlipperyEnvCfg(AntUnseenEnvCfg):
    terrain_friction: float = 0.3


@configclass
class AntUnseenStickyEnvCfg(AntUnseenEnvCfg):
    terrain_friction: float = 1.6


@configclass
class AntFlatFrictionEnvCfg(AntEnvCfg):
    """Original flat terrain with a different ground friction coefficient."""

    terrain_friction: float = 0.3

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.terrain.physics_material.static_friction = self.terrain_friction
        self.scene.terrain.physics_material.dynamic_friction = self.terrain_friction


@configclass
class AntFlatSlipperyEnvCfg(AntFlatFrictionEnvCfg):
    terrain_friction: float = 0.3


@configclass
class AntFlatStickyEnvCfg(AntFlatFrictionEnvCfg):
    terrain_friction: float = 1.6


@configclass
class AntRoughTerrainOnlyEnvCfg(AntRoughEnvCfg):
    """Ablation: rough terrain and curriculum only, no robot-side physics randomization."""

    def __post_init__(self):
        super().__post_init__()
        self.events.physics_material = None
        self.events.scale_torso_mass = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {}


@configclass
class AntFlatPhysicsOnlyEnvCfg(AntEnvCfg):
    """Ablation: original flat terrain with robot-side physics randomization only."""

    events: RoughEventCfg = RoughEventCfg()


@configclass
class AntRoughMildEnvCfg(AntRoughEnvCfg):
    """Rough terrain with a larger share of flat ground and milder obstacles, plus physics randomization."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ANT_ROUGH_MILD_TERRAINS_CFG


@configclass
class StableRewardsCfg(RewardsCfg):
    """Baseline rewards plus small penalties on body angular velocity and tilt."""

    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)


@configclass
class AntRoughAllEnvCfg(AntRoughEnvCfg):
    """Final-submission training: all 12 terrain types plus physics randomization."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ANT_ALL_TERRAINS_CFG


@configclass
class AntRoughAllStableEnvCfg(AntRoughAllEnvCfg):
    rewards: StableRewardsCfg = StableRewardsCfg()


@configclass
class AntUnseen2EnvCfg(AntUnseenEnvCfg):
    """Second held-out terrain set, disjoint from both training sets."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ANT_UNSEEN2_TERRAINS_CFG


##
# Variant with exteroceptive terrain perception
##


@configclass
class AntScanSceneCfg(AntRoughSceneCfg):
    terrain_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso",
        offset=RayCasterCfg.OffsetCfg(pos=(0.3, 0.0, 2.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.2, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class ScanObservationsCfg(ObservationsCfg):
    @configclass
    class ScanPolicyCfg(ObservationsCfg.PolicyCfg):
        height_scan = ObsTerm(
            func=mdp.terrain_height_scan,
            params={"sensor_cfg": SceneEntityCfg("terrain_scanner"), "offset": 0.5},
            clip=(-1.0, 1.0),
        )

    policy: ScanPolicyCfg = ScanPolicyCfg()


@configclass
class AntRoughAllScanEnvCfg(AntRoughAllEnvCfg):
    """All terrains, physics randomization and a forward-looking height scanner in the observation."""

    scene: AntScanSceneCfg = AntScanSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    observations: ScanObservationsCfg = ScanObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain_scanner.update_period = self.decimation * self.sim.dt


@configclass
class AntUnseenScanEnvCfg(AntUnseenEnvCfg):
    scene: AntScanSceneCfg = AntScanSceneCfg(num_envs=4096, env_spacing=5.0, clone_in_fabric=False)
    observations: ScanObservationsCfg = ScanObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain_scanner.update_period = self.decimation * self.sim.dt


@configclass
class AntUnseen2ScanEnvCfg(AntUnseenScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ANT_UNSEEN2_TERRAINS_CFG


@configclass
class AntFlatScanEnvCfg(AntUnseenScanEnvCfg):
    """Original flat plane for the scanner-equipped policy."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.terrain.physics_material.static_friction = self.terrain_friction
        self.scene.terrain.physics_material.dynamic_friction = self.terrain_friction


@configclass
class AntFlatScanSlipperyEnvCfg(AntFlatScanEnvCfg):
    terrain_friction: float = 0.3


@configclass
class AntFlatScanStickyEnvCfg(AntFlatScanEnvCfg):
    terrain_friction: float = 1.6


##
# Variant with a one-off penalty when the robot falls
##


@configclass
class DeathPenaltyRewardsCfg(RewardsCfg):
    """Baseline rewards plus a single penalty at the fall termination (about -10 per fall after dt scaling)."""

    fall_penalty = RewTerm(func=mdp.fall_termination, weight=-600.0)


@configclass
class AntRoughAllDeathEnvCfg(AntRoughAllEnvCfg):
    rewards: DeathPenaltyRewardsCfg = DeathPenaltyRewardsCfg()


@configclass
class AntRoughAllScanDeathEnvCfg(AntRoughAllScanEnvCfg):
    rewards: DeathPenaltyRewardsCfg = DeathPenaltyRewardsCfg()


##
# Evaluation terrain modelled on the TA's example image: a field of random-height grid boxes
##


@configclass
class AntGridEnvCfg(AntUnseenEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ANT_GRID_TERRAINS_CFG


@configclass
class AntGridSlipperyEnvCfg(AntGridEnvCfg):
    terrain_friction: float = 0.3


@configclass
class AntGridScanEnvCfg(AntUnseenScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ANT_GRID_TERRAINS_CFG


@configclass
class AntGridScanSlipperyEnvCfg(AntGridScanEnvCfg):
    terrain_friction: float = 0.3


##
# Training with a larger share of grid-box terrain and taller boxes (matches the example image)
##


@configclass
class AntRoughAllGridEnvCfg(AntRoughAllEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ANT_ALL_GRIDHEAVY_TERRAINS_CFG


@configclass
class AntRoughAllGridScanEnvCfg(AntRoughAllScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ANT_ALL_GRIDHEAVY_TERRAINS_CFG


##
# Diverse training set (pillars, steeper slopes, taller boxes) and a hard held-out test set
##


@configclass
class AntDiverseEnvCfg(AntRoughAllEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ANT_DIVERSE_TERRAINS_CFG


@configclass
class AntDiverseScanEnvCfg(AntRoughAllScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ANT_DIVERSE_TERRAINS_CFG


@configclass
class AntUnseen3HardEnvCfg(AntUnseenEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ANT_UNSEEN3_HARD_TERRAINS_CFG


@configclass
class AntUnseen3HardScanEnvCfg(AntUnseenScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ANT_UNSEEN3_HARD_TERRAINS_CFG


##
# PD joint-position control instead of direct torques
##

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab_assets.robots.ant import ANT_CFG

ANT_PD_CFG = ANT_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=40.0,
            damping=2.0,
            effort_limit_sim=15.0 * 7.5,
        ),
    },
)


@configclass
class PdActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class AntDiverseScanPdEnvCfg(AntDiverseScanEnvCfg):
    actions: PdActionsCfg = PdActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANT_PD_CFG


@configclass
class AntUnseen3HardScanPdEnvCfg(AntUnseen3HardScanEnvCfg):
    actions: PdActionsCfg = PdActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANT_PD_CFG


@configclass
class AntFlatScanPdEnvCfg(AntFlatScanEnvCfg):
    actions: PdActionsCfg = PdActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANT_PD_CFG


@configclass
class AntGridScanPdEnvCfg(AntGridScanEnvCfg):
    actions: PdActionsCfg = PdActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANT_PD_CFG


@configclass
class AntGridScanSlipperyPdEnvCfg(AntGridScanPdEnvCfg):
    terrain_friction: float = 0.3


@configclass
class AntUnseen2ScanPdEnvCfg(AntUnseen2ScanEnvCfg):
    actions: PdActionsCfg = PdActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANT_PD_CFG


##
# Observation history variant (last 3 observations stacked)
##


def _enable_history(cfg, length: int = 3):
    cfg.observations.policy.history_length = length
    cfg.observations.policy.flatten_history_dim = True


@configclass
class AntRoughAllScanHistEnvCfg(AntRoughAllScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _enable_history(self)


@configclass
class AntFlatScanHistEnvCfg(AntFlatScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _enable_history(self)


@configclass
class AntGridScanHistEnvCfg(AntGridScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _enable_history(self)


@configclass
class AntGridScanSlipperyHistEnvCfg(AntGridScanHistEnvCfg):
    terrain_friction: float = 0.3


@configclass
class AntUnseen2ScanHistEnvCfg(AntUnseen2ScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _enable_history(self)


@configclass
class AntUnseen3HardScanHistEnvCfg(AntUnseen3HardScanEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _enable_history(self)


##
# Final-submission evaluation entry points (TA: swap terrain / friction here)
##


@configclass
class AntFinalEvalEnvCfg(AntUnseen3HardScanHistEnvCfg):
    """Evaluation entry point for the submitted policy (height scanner + 3-step observation history).

    To evaluate on a new terrain, replace ``self.scene.terrain.terrain_generator`` with any
    ``TerrainGeneratorCfg`` (or set ``terrain_type="plane"`` and ``terrain_generator=None``), and set
    ``self.scene.terrain.physics_material.static_friction`` / ``dynamic_friction``. Rewards are the
    original Isaac-Ant-v0 terms, so the printed return is directly comparable to the baseline.
    """

    def __post_init__(self):
        super().__post_init__()
        # Example: self.scene.terrain.terrain_generator = ANT_GRID_TERRAINS_CFG
        # Example: self.scene.terrain.physics_material.static_friction = 0.5


@configclass
class AntFinalFlatEnvCfg(AntFlatScanHistEnvCfg):
    """Original flat plane for the submitted policy. Change ``terrain_friction`` for other coefficients."""

    terrain_friction: float = 1.0


@configclass
class AntRoughAllScanHistDeathEnvCfg(AntRoughAllScanHistEnvCfg):
    """Final configuration plus the one-off fall penalty (safety-first priority)."""

    rewards: DeathPenaltyRewardsCfg = DeathPenaltyRewardsCfg()


##
# H6: smooth, grounded gait as a human priority
##


@configclass
class SmoothRewardsCfg(RewardsCfg):
    """Baseline rewards plus penalties on vertical bounce, body rotation, action jerk and joint acceleration."""

    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)


@configclass
class AntRoughAllScanHistSmoothEnvCfg(AntRoughAllScanHistEnvCfg):
    rewards: SmoothRewardsCfg = SmoothRewardsCfg()


ANT_PD_SOFT_CFG = ANT_CFG.replace(
    prim_path="{ENV_REGEX_NS}/Robot",
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=8.0,
            damping=0.3,
            effort_limit_sim=15.0 * 7.5,
        ),
    },
)


@configclass
class AntRoughAllScanHistPdSoftEnvCfg(AntRoughAllScanHistEnvCfg):
    actions: PdActionsCfg = PdActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANT_PD_SOFT_CFG


@configclass
class AntUnseen3HardScanHistPdSoftEnvCfg(AntUnseen3HardScanHistEnvCfg):
    actions: PdActionsCfg = PdActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANT_PD_SOFT_CFG


@configclass
class AntFlatScanHistPdSoftEnvCfg(AntFlatScanHistEnvCfg):
    actions: PdActionsCfg = PdActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = ANT_PD_SOFT_CFG


##
# H7: grounded gait - strong bounce penalties, terrain-relative height tracking, optional lower torque limit
##


@configclass
class GroundedRewardsCfg(RewardsCfg):
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-4.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-7)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-5.0,
        params={"target_height": 0.5, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )


@configclass
class LowTorqueActionsCfg:
    joint_effort = mdp.JointEffortActionCfg(asset_name="robot", joint_names=[".*"], scale=4.0)


@configclass
class AntRoughAllScanHistGroundedEnvCfg(AntRoughAllScanHistEnvCfg):
    rewards: GroundedRewardsCfg = GroundedRewardsCfg()


@configclass
class AntRoughAllScanHistGroundedLowTorqueEnvCfg(AntRoughAllScanHistGroundedEnvCfg):
    actions: LowTorqueActionsCfg = LowTorqueActionsCfg()


@configclass
class AntFlatScanHistLowTorqueEnvCfg(AntFlatScanHistEnvCfg):
    actions: LowTorqueActionsCfg = LowTorqueActionsCfg()


@configclass
class AntGridScanHistLowTorqueEnvCfg(AntGridScanHistEnvCfg):
    actions: LowTorqueActionsCfg = LowTorqueActionsCfg()


@configclass
class AntGridScanSlipperyHistLowTorqueEnvCfg(AntGridScanSlipperyHistEnvCfg):
    actions: LowTorqueActionsCfg = LowTorqueActionsCfg()


@configclass
class AntUnseen2ScanHistLowTorqueEnvCfg(AntUnseen2ScanHistEnvCfg):
    actions: LowTorqueActionsCfg = LowTorqueActionsCfg()


@configclass
class AntUnseen3HardScanHistLowTorqueEnvCfg(AntUnseen3HardScanHistEnvCfg):
    actions: LowTorqueActionsCfg = LowTorqueActionsCfg()


##
# Stress tests on the hard held-out terrain: extreme friction, bouncy ground, random start heading
##

import math


def _stress(cfg, friction=None, restitution=None, yaw=None):
    if friction is not None:
        cfg.scene.terrain.physics_material.static_friction = friction
        cfg.scene.terrain.physics_material.dynamic_friction = friction
    if restitution is not None:
        cfg.scene.terrain.physics_material.restitution = restitution
    if yaw is not None:
        cfg.events.reset_base.params["pose_range"] = {"yaw": (-yaw, yaw)}


@configclass
class AntStressFriction01EnvCfg(AntUnseen3HardScanHistEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _stress(self, friction=0.1)


@configclass
class AntStressFriction20EnvCfg(AntUnseen3HardScanHistEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _stress(self, friction=2.0)


@configclass
class AntStressBouncyEnvCfg(AntUnseen3HardScanHistEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _stress(self, restitution=0.5)


@configclass
class AntStressYawEnvCfg(AntUnseen3HardScanHistEnvCfg):
    def __post_init__(self):
        super().__post_init__(); _stress(self, yaw=math.pi)


@configclass
class AntStressFriction01LowTorqueEnvCfg(AntStressFriction01EnvCfg):
    actions: LowTorqueActionsCfg = LowTorqueActionsCfg()


@configclass
class AntStressFriction20LowTorqueEnvCfg(AntStressFriction20EnvCfg):
    actions: LowTorqueActionsCfg = LowTorqueActionsCfg()


@configclass
class AntStressBouncyLowTorqueEnvCfg(AntStressBouncyEnvCfg):
    actions: LowTorqueActionsCfg = LowTorqueActionsCfg()


@configclass
class AntStressYawLowTorqueEnvCfg(AntStressYawEnvCfg):
    actions: LowTorqueActionsCfg = LowTorqueActionsCfg()


##
# Final candidates: diverse terrain (incl. pillars, steep slopes, tall boxes) + scanner + history, baseline PPO
##


@configclass
class AntFinalTrainEnvCfg(AntRoughAllScanHistEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ANT_FINAL_TERRAINS_CFG


@configclass
class AntFinalTrainSmoothEnvCfg(AntFinalTrainEnvCfg):
    rewards: SmoothRewardsCfg = SmoothRewardsCfg()


@configclass
class AntFinalTrainGroundedEnvCfg(AntFinalTrainEnvCfg):
    rewards: GroundedRewardsCfg = GroundedRewardsCfg()


@configclass
class AntFinalTrainGroundedLowTorqueEnvCfg(AntFinalTrainGroundedEnvCfg):
    actions: LowTorqueActionsCfg = LowTorqueActionsCfg()


@configclass
class AntUnseen4ScanHistEnvCfg(AntUnseen3HardScanHistEnvCfg):
    """Held-out obstacle course: tall boxes and cones to steer around, wave and steep slopes."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = ANT_UNSEEN4_OBSTACLES_TERRAINS_CFG


@configclass
class AntUnseen4ScanHistLowTorqueEnvCfg(AntUnseen4ScanHistEnvCfg):
    actions: LowTorqueActionsCfg = LowTorqueActionsCfg()


@configclass
class GroundedLiteRewardsCfg(RewardsCfg):
    """Half-strength grounded penalties: between SmoothRewardsCfg and GroundedRewardsCfg."""

    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-7)
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.0,
        params={"target_height": 0.5, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )


@configclass
class AntFinalTrainGroundedLiteEnvCfg(AntFinalTrainEnvCfg):
    rewards: GroundedLiteRewardsCfg = GroundedLiteRewardsCfg()


@configclass
class AntFinalObstaclesEnvCfg(AntUnseen4ScanHistEnvCfg):
    """Evaluation entry point on the held-out obstacle course (tall boxes and cones to steer around)."""
