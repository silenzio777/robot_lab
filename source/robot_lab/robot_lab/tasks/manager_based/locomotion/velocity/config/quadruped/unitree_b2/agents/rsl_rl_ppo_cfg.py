# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeB2RoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_b2_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeB2FlatPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "unitree_b2_flat"


@configclass
class UnitreeB2CrawlPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # Own experiment_name -> own logs/rsl_rl/unitree_b2_crawl/ directory, so this
        # fine-tune run's checkpoints never land in (or get confused with) the ongoing
        # unitree_b2_rough walking run's own log folder. Same network architecture/obs/
        # action dims as UnitreeB2RoughPPORunnerCfg (unchanged), so a walking checkpoint's
        # weights load into this run's model with no shape mismatch -- see
        # crawl_env_cfg.py's own docstring and train_research/robot_lab/README.md for the
        # actual warm-start recipe (seed the new experiment dir with the source
        # checkpoint, then --resume --load_run <that dir> --checkpoint <file>).
        self.experiment_name = "unitree_b2_crawl"


@configclass
class UnitreeB2RearStandPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # Own logs/rsl_rl/unitree_b2_rear_stand/ directory; same network/obs/action
        # shapes as the rough walking runner, so a walking checkpoint can warm-start
        # this run if desired (same recipe as crawl's own note above).
        self.experiment_name = "unitree_b2_rear_stand"


@configclass
class UnitreeB2JumpPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # Same reasoning as UnitreeB2CrawlPPORunnerCfg's own experiment_name split --
        # own logs/rsl_rl/unitree_b2_jump/ directory, warm-started from the same
        # unitree_b2_rough walking checkpoint (see jump_env_cfg.py's own docstring).
        self.experiment_name = "unitree_b2_jump"


@configclass
class UnitreeB2VisionPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # Own logs/rsl_rl/unitree_b2_vision/ directory -- unlike Crawl/Jump this can't
        # warm-start from the walking checkpoint anyway (see vision_env_cfg.py's own
        # docstring: the observation shape itself differs once height_scan is back on),
        # so there's no seed-checkpoint step for this one, just a from-scratch run.
        self.experiment_name = "unitree_b2_vision"
