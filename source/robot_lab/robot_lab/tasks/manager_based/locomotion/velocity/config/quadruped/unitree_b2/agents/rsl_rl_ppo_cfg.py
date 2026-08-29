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
class UnitreeB2WalkPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # Own logs/rsl_rl/unitree_b2_walk/ directory -- same network/obs/action shapes
        # as stock rough (walk_env_cfg.py only turns on gait-rhythm reward weights,
        # nothing structural), so a stock-rough walking checkpoint can warm-start this
        # run (same recipe as crawl/rear_stand's own note above).
        self.experiment_name = "unitree_b2_walk"

        # entropy_coef 0.01 -> 0.005 (2026-08-26, base's diagnosis): the
        # base_height_l2=-16.0 jitter-free control hit a sustained Loss/
        # learning_rate floor-pin (200+ iterations, no recovery) with
        # PROGRESSIVELY worsening Train/mean_reward (not a single-step spike
        # with immediate recovery -- the two earlier benign floor-touches in
        # this same run recovered within ~90 iterations with reward staying
        # healthy throughout) at it29280+, ~2100 iterations into the run --
        # base recognized this as the SAME signature already diagnosed and
        # fixed three times elsewhere in this project: jump (2026-08-08,
        # noise_std climbed and never reconverged), rear_stand (2026-08-10,
        # sustained action_rate_l2 instability 1200+ iterations), leg_lift
        # (2026-08-13, worst single-iteration reward growing in severity
        # across consecutive checks) -- see each cfg's own comment below.
        # WALK was the one B2 skill in this family that never got this fix.
        # Walk-only override -- doesn't touch rough/crawl/rear_stand/jump/
        # leg_lift/vision's own entropy_coef.
        self.algorithm.entropy_coef = 0.005


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
class UnitreeB2HandstandPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # Own logs/rsl_rl/unitree_b2_handstand/ directory. Default entropy_coef
        # (0.01, inherited) kept as-is for this first pass -- NOT copying rear_
        # stand's own 0.005 override, since that fix was diagnosed against a
        # SPECIFIC observed action_rate_l2 instability on rear_stand's own
        # training run, not a general rule for "any 2-leg-support skill". If
        # the same instability signature (sustained value_function loss blowup,
        # 1000+ iterations, not self-resolving) shows up here too, this is the
        # first thing to check -- see UnitreeB2RearStandPPORunnerCfg's own
        # comment for the exact diagnostic signature to compare against.
        self.experiment_name = "unitree_b2_handstand"


@configclass
class UnitreeB2RearStandPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # Own logs/rsl_rl/unitree_b2_rear_stand/ directory; same network/obs/action
        # shapes as the rough walking runner, so a walking checkpoint can warm-start
        # this run if desired (same recipe as crawl's own note above).
        self.experiment_name = "unitree_b2_rear_stand"

        # entropy_coef 0.01 -> 0.005 (2026-08-10 night, autonomous fix under explicit
        # nightly authorization): the v4 resume run (2026-08-10_03-35-39, from
        # model_9000) hit a sustained action_rate_l2 instability at it18694-19894+ --
        # 1200+ iterations, 86% with value_function loss >1000 (peak 15.6M, worst
        # single iteration reward -137149) -- qualitatively different from every
        # earlier spike that night (which self-resolved within 200-400 iterations).
        # episode_length/orientation_tracking never degraded even at the worst point
        # (same "isolated action_rate_l2 artifact, not real policy collapse" signature
        # as jump 08-08), but the DURATION crossed into "not self-resolving on its
        # own" territory. Same fix that solved it for jump (see
        # UnitreeB2JumpPPORunnerCfg's own comment) -- calms noise_std, which reduces
        # the extreme rare actions that trigger action_rate_l2's own unbounded blow-up.
        # Rear_stand-only override -- doesn't touch rough/crawl/jump/vision.
        self.algorithm.entropy_coef = 0.005


@configclass
class UnitreeB2RearStandStageAPPORunnerCfg(UnitreeB2RearStandPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # Own logs/rsl_rl/unitree_b2_rear_stand_stage_a/ directory -- same split
        # reasoning as crawl/jump's own experiment_name overrides above: Stage A
        # (3-slot command, see rear_stand_env_cfg.py's RearStandStageACommand)
        # is a genuinely different network input shape than the full 6-slot
        # RearStand task, so its checkpoints must never land in (or get
        # confused with) unitree_b2_rear_stand/'s own directory. Inherits the
        # parent's entropy_coef=0.005 (same reward economy minus 2 terms, same
        # stability profile expected) rather than reverting to Rough's default.
        self.experiment_name = "unitree_b2_rear_stand_stage_a"


@configclass
class UnitreeB2JumpPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # Same reasoning as UnitreeB2CrawlPPORunnerCfg's own experiment_name split --
        # own logs/rsl_rl/unitree_b2_jump/ directory, warm-started from the same
        # unitree_b2_rough walking checkpoint (see jump_env_cfg.py's own docstring).
        self.experiment_name = "unitree_b2_jump"

        # entropy_coef 0.01 -> 0.005 (2026-08-08, overnight instability): noise_std
        # climbed from ~1.6 to 2.2-2.9 over the night and never reconverged --
        # unusually high and sustained for a task with a 1.0 init. A repeating
        # pattern of action_rate_l2 blowing up (up to -1.2M on isolated iterations,
        # occasionally correlating with REAL jump_direction_velocity/flight_distance
        # degradation, not just log noise) tracked with these high-noise stretches.
        # Halving the entropy bonus should let exploration settle instead of
        # perpetually re-inflating after every improvement. Jump-only override --
        # doesn't touch rough/crawl/rear_stand/vision's own entropy_coef.
        self.algorithm.entropy_coef = 0.005


@configclass
class UnitreeB2JumpV10PPORunnerCfg(UnitreeB2JumpPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # Separate logs/rsl_rl/unitree_b2_jump_v10/ directory -- the v10 line
        # (2026-08-26, owner's from-scratch minimal redesign) is a genuinely
        # different gym task (jump_v10_env_cfg.py, not jump_env_cfg.py) and
        # warm-starts from stage_a_standing, not any prior jump-line
        # checkpoint -- keeping it in the OLD unitree_b2_jump directory would
        # mix incompatible checkpoint histories (different reward economy,
        # different command semantics) under one experiment name.
        self.experiment_name = "unitree_b2_jump_v10"

        # 0.01 -> 0.005, REVERTED back (2026-08-27 evening, base's diagnosis,
        # empirically confirmed): the 0.01 restoration below (same day,
        # earlier) was meant to help ignition exploration, but it did the
        # opposite -- noise_std climbed to a ~1.7 plateau, and a direct
        # stochastic-noise probe (jump_v10_stochastic_idle_probe.py) proved
        # this reliably trips illegal_contact-magnitude forces (1000s of N)
        # on base_link/hip/thigh during idle/crouch, across every tested
        # seed. Since launch_active_ratio measured EXACTLY 0.0000 across the
        # ENTIRE run's log (6440/6440 samples) -- launch was never once
        # reached in a training rollout -- the raised entropy never even got
        # to explore the launch phase it was raised for; it only spent its
        # extra noise budget killing episodes in idle/crouch, the one phase
        # already warm-started and not in need of exploration. Reverting to
        # the same 0.005 already proven correct for jump-v3/rear_stand/
        # leg_lift's own version of this exact instability signature. Paired
        # with idle_time_range (2,4)->(0.75,1.5) in jump_v10_env_cfg.py (same
        # diagnosis, complementary fix -- see that file's own comment).
        #
        # ORIGINAL comment (2026-08-27 morning, now superseded, kept for
        # history): this class inherits from UnitreeB2JumpPPORunnerCfg,
        # which halves entropy_coef 0.01->0.005 for a completely different
        # reason (2026-08-08 overnight action_rate_l2 instability on the OLD
        # jump-v3 economy) -- jump_v10 silently carried that halved value
        # despite never having that instability, and despite the ignition
        # problem (see jump_v10_launch_calf_extend's own docstring) needing
        # MORE exploration, not less, right when it matters. Restored to
        # rough's own stock 0.01 -- jump_v10-specific override, doesn't
        # touch the old jump line's own tuned 0.005.
        self.algorithm.entropy_coef = 0.005


@configclass
class UnitreeB2LegLiftPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # Own logs/rsl_rl/unitree_b2_leg_lift/ directory; same network/obs/action
        # shapes as the rough walking runner (leg_lift_env_cfg.py deliberately kept
        # the command at 3 slots), so a walking checkpoint can warm-start this run if
        # desired (same recipe as crawl/rear_stand's own note above).
        self.experiment_name = "unitree_b2_leg_lift"
        # 8000 -> 20000 (2026-08-11, first bench test: at 8000 only ONE of four legs
        # (RR) had begun responding at all -- the budget ran out mid-learning, not
        # post-convergence; the front lifts need a trained CoM shift backward on top
        # of everything else, which is exactly the expensive part).
        self.max_iterations = 20000

        # entropy_coef 0.01 -> 0.005 (2026-08-13, bench-monitor autonomy fix, v3b
        # gated-height run): the SAME isolated action_rate_l2/vloss blow-up pattern
        # already fixed this way for jump (2026-08-08) and rear_stand v4 (2026-08-10)
        # -- reward/episode_length recover on their own within ~100-300 iterations
        # each time (elen never dropped below 1000 across either episode, confirmed
        # line-by-line), but the pattern RECURRED and grew worse across two
        # consecutive 30-min monitoring checks (worst single-iteration reward -111 ->
        # -231.6, worst vloss 6.79 -> 8.52) rather than staying at a stable severity.
        # Leg-lift was the one task in this family that never got this fix. Halving
        # the entropy bonus should let exploration settle instead of repeatedly
        # re-triggering these spikes. Leg-lift-only override -- doesn't touch rough/
        # crawl/vision's own entropy_coef.
        self.algorithm.entropy_coef = 0.005


@configclass
class UnitreeB2VisionPPORunnerCfg(UnitreeB2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        # Own logs/rsl_rl/unitree_b2_vision/ directory -- unlike Crawl/Jump this can't
        # warm-start from the walking checkpoint anyway (see vision_env_cfg.py's own
        # docstring: the observation shape itself differs once height_scan is back on),
        # so there's no seed-checkpoint step for this one, just a from-scratch run.
        self.experiment_name = "unitree_b2_vision"
