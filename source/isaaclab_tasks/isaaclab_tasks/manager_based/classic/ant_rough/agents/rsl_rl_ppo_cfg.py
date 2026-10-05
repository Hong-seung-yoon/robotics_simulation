from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg


@configclass
class AntRoughPPORunnerCfg(AntPPORunnerCfg):
    max_iterations = 1500
    experiment_name = "ant_rough"


@configclass
class AntAblationPPORunnerCfg(AntRoughPPORunnerCfg):
    experiment_name = "ant_ablation"


@configclass
class AntRoughExplorePPORunnerCfg(AntRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.algorithm.entropy_coef = 0.005
