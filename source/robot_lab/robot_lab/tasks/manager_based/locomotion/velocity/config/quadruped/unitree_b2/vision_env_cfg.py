# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import torch

import isaaclab.terrains as terrain_gen
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeB2RoughEnvCfg


def vision_base_height_l2(env, target_height: float, asset_cfg=None, sensor_cfg=None) -> torch.Tensor:
    """Per-env-guarded replacement for mdp.base_height_l2 -- same math, different
    invalid-ray handling. The stock term's guard is GLOBAL: one inf ray anywhere in
    any env makes it substitute the target with the robot's own current height,
    zeroing the penalty for ALL envs that tick. On void-bearing terrain (gaps/
    stairs_holes have no floor -- rays into them MISS and return float('inf'), warp
    ops' documented miss value) some env always has a void ray, so the height anchor
    was silently dead 100% of the time (Episode_Reward/base_height_l2 pinned at
    exactly 0.0 in run 2026-08-05_09-26-46 while the cfg weight was -8.0). Same bug
    class -- and same per-env fix -- as base_height_below_scan above."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    sensor = env.scene[sensor_cfg.name if sensor_cfg is not None else "height_scanner_base"]
    ray_z = sensor.data.ray_hits_w[..., 2]
    finite = torch.isfinite(ray_z) & (ray_z.abs() < 1e6)
    n_valid = finite.sum(dim=1)
    ground = torch.where(finite, ray_z, torch.zeros_like(ray_z)).sum(dim=1) / n_valid.clamp(min=1)
    err = torch.square(asset.data.root_pos_w[:, 2] - (target_height + ground))
    err = err * (n_valid > 0).float()
    # Same uprightness modulation as the stock term (don't punish height while tipped).
    err = err * (torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7)
    return err


def base_height_below_scan(env, threshold: float, asset_cfg=None, sensor_cfg=None) -> torch.Tensor:
    """Terminate when the base rides lower than `threshold` above the terrain surface
    measured by the height_scanner (mean of its ray hits -- same sensor-adjusted-height
    idiom as this repo's own `mdp.base_height_l2`).

    Exists because the first run on the vision-forcing terrain (2026-08-03_23-33-13)
    had NO termination that fires when a robot drops INTO a gap/stone-hole/pit and
    wedges there: base never touches (legs wedge first), orientation stays legal, so
    the episode ground out its full 1000 steps of physics-thrashing penalties
    (lin_vel_z_l2 ~10x normal) while the terrain curriculum read the wreckage as
    "everything fails" and pinned all envs to level 0 (terrain_levels collapsed
    2.7 -> 0.03, tracking ~0.07 -- see train_research/TRAIN_RESEARCH.md).

    World-Z thresholds can't express this on generated terrain (descending a pyramid
    stair legitimately takes the base meters below its env origin) -- but height above
    the LOCAL surface can: a walking B2 rides ~0.55m above it, a robot wedged in a
    hole sits near 0. The scan mean is also naturally conservative at a hole's edge:
    rays hitting the surrounding stone tops keep the mean high, so a body dropped
    below them triggers immediately, while a robot merely LOOKING over a pit edge
    (mean dips, measured height grows) never does.
    """
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    sensor = env.scene[sensor_cfg.name if sensor_cfg is not None else "height_scanner"]
    ray_hits = sensor.data.ray_hits_w[..., 2]
    # Invalid-ray guard is PER-ENV, deliberately NOT the global any()-check
    # mdp.base_height_l2 uses -- found the hard way (run 2026-08-04_00-22-37): rays
    # that MISS return float('inf') (warp ops' documented miss value), and with
    # gap/hole terrains SOME env somewhere always has a ray pointed into a void, so
    # a global "any inf anywhere -> skip everything" guard permanently disabled this
    # termination (0 firings ever, wedged robots thrashing out full 1000-step
    # episodes again -- observable as the same lin_vel_z_l2 spikes the termination
    # exists to kill). A reward term can afford a global skip for one tick; a
    # termination can't. Per-env: average each env's own valid rays; an env with no
    # valid rays at all just doesn't terminate this tick.
    finite = torch.isfinite(ray_hits) & (ray_hits.abs() < 1e6)
    n_valid = finite.sum(dim=1)
    ground = torch.where(finite, ray_hits, torch.zeros_like(ray_hits)).sum(dim=1) / n_valid.clamp(min=1)
    below = (asset.data.root_pos_w[:, 2] - ground) < threshold
    return below & (n_valid > 0)

# Vision-forcing terrain set (2026-08-03). The first from-scratch vision run (~8.8k
# iterations on the stock ROUGH_TERRAINS_CFG) plateaued at reward ~210 by it2200 and
# gained only ~6% over the following 6600 iterations -- because the stock terrain
# (stairs <=0.23m, boxes, slopes) is entirely solvable BLIND by proprioception alone,
# so the network has almost no gradient pressure to ever read the height_scan input.
# This set replaces most of it with terrain classes that are impossible (or heavily
# failure-prone) without knowing WHERE to step, while keeping the curriculum's easy
# end genuinely easy (difficulty scales every range from its first element):
#   - gaps          : a void ring around the platform, 0.1->0.55m wide -- a blind
#                     robot walks a leg straight into it, a seeing one steps across.
#   - stepping_stones: stones shrink (1.2->0.35m) and spread apart (0.05->0.35m) with
#                     difficulty, floor between them dropped 0.6m -- the canonical
#                     "must look where you step" terrain. holes_depth is -0.6 rather
#                     than the -10 default so a missed step lands and terminates the
#                     episode promptly instead of falling into a bottomless render.
#   - stairs_holes  : pyramid stairs where only a platform_width-wide strip has steps
#                     at all (holes=True), the rest is a drop -- straying off the
#                     strip blind is a fall.
#   - pyramid stairs (both ways) raised to 0.35m top-end (was 0.23 -- comfortably
#                     blind-walkable for a robot this size).
#   - pit           : 0.1->0.45m deep pit with vertical edges to see and negotiate.
#   - random_rough  : kept as the locomotion-quality anchor (proprio-solvable).
# Global generator params (size/rows/cols/scales) identical to ROUGH_TERRAINS_CFG.
# curriculum=True set HERE because the base VelocityEnvCfg.__post_init__ sets that
# flag on the generator it already has -- ours replaces it after that ran.
B2_VISION_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # Easy ends retuned 2026-08-04 after the first run collapsed to level 0 and
        # STILL failed there: at difficulty 0 the gap was 0.1m and the stone slots
        # 0.05m -- exactly foot-width traps, so even the easiest row wedged legs.
        # Now difficulty 0 is honestly trivial: a 2cm crack no foot fits into, and
        # stones that touch (distance 0.0 = solid floor); the hard ends unchanged.
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.15,
            gap_width_range=(0.02, 0.55),
            platform_width=2.5,
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.15,
            stone_width_range=(0.35, 1.2),
            stone_distance_range=(0.0, 0.35),
            stone_height_max=0.05,
            holes_depth=-0.6,
            platform_width=2.0,
            border_width=0.25,
        ),
        "stairs_holes": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.1,
            step_height_range=(0.05, 0.25),
            step_width=0.35,
            platform_width=2.5,
            border_width=1.0,
            holes=True,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.35),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.05, 0.35),
            step_width=0.35,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.1,
            pit_depth_range=(0.1, 0.45),
            platform_width=2.5,
        ),
        # 0.15 -> 0.2 (taking the 0.05 dropped from gaps): the proprio-solvable
        # anchor got heavier after the first run showed the easy rows must carry
        # enough plain-walkable ground for basic locomotion to bootstrap at all.
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2,
            noise_range=(0.02, 0.10),
            noise_step=0.02,
            border_width=0.25,
        ),
    },
)


@configclass
class UnitreeB2VisionRoughEnvCfg(UnitreeB2RoughEnvCfg):
    """Same gait/reward recipe as the plain walking UnitreeB2RoughEnvCfg -- the ONLY
    change is re-enabling the `height_scan` observation that the parent class turns off
    (`self.observations.policy.height_scan = None`, see rough_env_cfg.py). Everything
    else (rewards, events, terrain curriculum) is deliberately untouched, so this is a
    clean ablation: does giving the policy the same terrain-elevation grid the critic
    already always sees (CriticCfg.height_scan is never nulled, only PolicyCfg's is --
    a standard asymmetric-actor-critic setup) let it anticipate a step/threshold instead
    of only reacting to it via contact, with nothing else in the training recipe
    changed to compensate.

    `height_scanner` (the RayCaster feeding this observation, `size=[1.6, 1.0]` at
    `resolution=0.1` -- see velocity_env_cfg.py's own SceneCfg) is already wired in
    at the base class level and untouched here.

    2026-08-03 update: no longer a pure ablation -- the terrain generator is ALSO
    replaced with B2_VISION_TERRAINS_CFG (above), after the first from-scratch run
    proved the stock ROUGH_TERRAINS_CFG never forces the policy to actually read
    the scan (see the constant's own comment and train_research/TRAIN_RESEARCH.md).

    NOT warm-startable from a plain walking checkpoint (model_5000.pt) the way
    crawl/jump are -- those only change reward weights (same obs/action shape), this
    changes the policy's own observation dimension (+height_scan's own ~150-190 grid
    cells on top of the base ~45), so the network's first layer shape itself differs.
    Trains from scratch.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Vision-forcing terrain (see B2_VISION_TERRAINS_CFG's own comment). Assigned
        # AFTER super().__post_init__() on purpose: the base class's own post-init
        # touches the generator it had at the time (e.g. sets .curriculum from the
        # curriculum manager's state), so ours carries curriculum=True in its own
        # definition rather than relying on that already-finished pass.
        self.scene.terrain.terrain_generator = B2_VISION_TERRAINS_CFG

        # "Wedged in a hole" termination (see base_height_below_scan's own docstring
        # for the failure mode this kills). New attribute on the configclass instance
        # is picked up by the termination manager the same way declared terms are
        # (ManagerBase iterates instance __dict__).
        #
        # Threshold history: 0.28 in the first fixed run (2026-08-04_00-30-22) --
        # which trained fine but converged to a CROUCHED walk: bench-measured base
        # height 0.34m at it15000, byte-for-byte crawl height ("собака ползает на
        # коленях", confirmed by direct measurement, walk=0.551 / vision=0.342 /
        # crawl=0.345). Cause: crouching lowers the CoM and wins on hole-terrain
        # stability, 0.34 was comfortably legal under a 0.28 threshold, and NOTHING
        # in the recipe pushed back -- base_height_l2 is weight 0 in the walking
        # recipe (height emerges from dynamics on stock terrain, but is a free
        # variable the moment the terrain rewards sinking it). Raised to 0.35: crawl-
        # height walking is now lethal, while genuine stair-descent transients
        # (scan-relative height ~0.45+) stay clear.
        self.terminations.base_height_below_scan = DoneTerm(
            func=base_height_below_scan,
            params={
                "threshold": 0.35,
                "asset_cfg": SceneEntityCfg("robot"),
                "sensor_cfg": SceneEntityCfg("height_scanner"),
            },
        )

        # ...and the positive side of the same fix: actively hold standing height.
        # The crawl variant already validated this exact term as THE height driver
        # (weight -8.0, target 0.35 -> bench-measured 0.345, within 5mm of target);
        # same magnitude here, at the recipe's own standing target (0.53, already
        # configured by the parent -- only the weight was 0). Its sensor_cfg is
        # `height_scanner_base` (the small under-base scanner, untouched by this
        # config), so the target is terrain-relative on rough ground.
        self.rewards.base_height_l2.weight = -8.0
        # Stock func swapped for the per-env-guarded version -- see its docstring;
        # params (asset_cfg/sensor_cfg/target_height) are signature-compatible.
        self.rewards.base_height_l2.func = vision_base_height_l2

        # Undo the parent's own `self.observations.policy.height_scan = None` -- same
        # ObsTerm the parent class's PolicyCfg (and CriticCfg, which never loses it)
        # defines in velocity_env_cfg.py, re-created here since the parent's __post_init__
        # already ran and nulled the field on this instance.
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        # UnitreeB2RoughEnvCfg's own disable_zero_weight_rewards() call is gated on
        # self.__class__.__name__ == "UnitreeB2RoughEnvCfg" (see rough_env_cfg.py), so
        # it does NOT fire for this subclass -- every zero-weight scaffolding term
        # (e.g. wheel_vel_penalty, dormant for non-wheeled robots, joint_names="")
        # stays a real RewTerm instead of being turned into None, and the reward
        # manager crashes trying to resolve its empty joint_names against B2's actual
        # joints. crawl_env_cfg.py/jump_env_cfg.py both already repeat this same call
        # with their own class name -- this was missing here, the actual bug behind
        # the "Not all regular expressions are matched ... wheel_vel_penalty" crash.
        if self.__class__.__name__ == "UnitreeB2VisionRoughEnvCfg":
            self.disable_zero_weight_rewards()
