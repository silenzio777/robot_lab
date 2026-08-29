# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Ramp traversal up to 37 degrees for B2 -- the OTHER half of the owner's
2026-08-30 request (`extreme-parkour`, CMU/ICRA2024, tilted-ramp walking up to
37deg, blind/proprioception-only -- see `handstand_env_cfg.py`'s own module
docstring for the shared context, this file covers the unrelated skill).

Unlike handstand, this is NOT a leg-support-topology change -- it's ordinary
quadruped walking, just on steep terrain. Built as a terrain-only extension of
WALK's OWN already-live reward economy (`walk_env_cfg.py`'s phase-clock/gait-
rhythm work), not a new economy from scratch -- same "one variable at a time"
discipline `vision_env_cfg.py`'s own B2_VISION_TERRAINS_CFG comment documents
for its own terrain-only fork off the stock rough config.

Terrain: IsaacLab's stock `HfPyramidSlopedTerrainCfg`/`HfInvertedPyramidSlopedTerrainCfg`
(confirmed reading `terrains/config/rough.py` + `height_field/hf_terrains_cfg.py`
directly -- `slope_range` is in RADIANS, not a rise/run ratio). Stock
ROUGH_TERRAINS_CFG only goes to slope_range=(0.0,0.4) (~22.9deg) -- 37deg =
0.6458 rad needs its own generator, not a stock import.

KNOWN, FLAGGED, NOT PRE-EMPTIVELY "FIXED" RISK: WALK's own inherited `upward`
reward (`mdp.rewards.upward`, read directly: `square(1 - projected_gravity_b[:,2])`)
rewards the body staying WORLD-flat regardless of terrain slope -- at 37deg,
meaningfully steeper than what this term has ever been exercised against in
this repo (stock library's own max is 22.9deg), a real quadruped naturally
pitches somewhat WITH the slope for CoM balance, which this term would
directly fight. Deliberately NOT re-weighted here without evidence -- same
"first guess, calibrate from what training produces" discipline every other
first-pass constant in this codebase's history follows (guessing a new number
with no measurement behind it is not lower-risk than leaving the loudly-
inherited one alone). If training shows the diagnostic signature "flattens on
steep terrain instead of climbing, or tips over past ~20-25deg rows" --
`self.rewards.upward.weight` is the first thing to check, not something else.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

from .walk_env_cfg import UnitreeB2WalkRoughEnvCfg

MAX_RAMP_ANGLE_DEG = 37.0  # extreme-parkour's own reported result, the owner's explicit target
MAX_RAMP_ANGLE_RAD = MAX_RAMP_ANGLE_DEG * 3.14159265 / 180.0  # 0.6458 rad

B2_RAMP_TERRAINS_CFG = TerrainGeneratorCfg(
    # Same size/border/resolution convention as ROUGH_TERRAINS_CFG/B2_VISION_TERRAINS_CFG
    # (not re-derived -- no reason to guess a new grid scale for this robot).
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,  # row 0 near-flat, row 9 approaches MAX_RAMP_ANGLE_RAD -- same progressive-difficulty idiom as every other terrain generator in this repo
    sub_terrains={
        # Up-slope and down-slope are genuinely different challenges (different
        # failure modes -- climbing traction vs. descent control), both needed,
        # same reasoning B2_VISION_TERRAINS_CFG's own up/down-facing pairs use.
        "ramp_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.45,
            slope_range=(0.0, MAX_RAMP_ANGLE_RAD),
            platform_width=2.0,
            border_width=0.25,
        ),
        "ramp_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.45,
            slope_range=(0.0, MAX_RAMP_ANGLE_RAD),
            platform_width=2.0,
            border_width=0.25,
        ),
        # Small share of ordinary rough-noise terrain -- grounding against
        # overfitting to perfectly smooth pyramid ramps specifically, same
        # modest-mix idiom ROUGH_TERRAINS_CFG itself uses (not 100% one type).
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
    },
)


@configclass
class UnitreeB2RampWalkEnvCfg(UnitreeB2WalkRoughEnvCfg):
    """WALK's own gait-rhythm/phase-clock economy, unchanged -- only the
    terrain generator differs from stock UnitreeB2WalkRoughEnvCfg."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = B2_RAMP_TERRAINS_CFG

        # FIX (train, 2026-08-30, found at launch): UnitreeB2WalkRoughEnvCfg's own
        # disable_zero_weight_rewards() call is gated on
        # self.__class__.__name__ == "UnitreeB2WalkRoughEnvCfg" (walk_env_cfg.py),
        # so it does NOT fire for this subclass -- every zero-weight scaffolding
        # term (e.g. wheel_vel_penalty, dormant for non-wheeled robots,
        # joint_names="") stays a real RewTerm instead of being turned into None,
        # and the reward manager crashes trying to resolve its empty joint_names
        # against B2's actual joints ("Not all regular expressions are matched").
        # Same class of bug already documented+fixed in vision_env_cfg.py and
        # jump_v10_env_cfg.py's own subclasses -- missing here, not a new mechanism.
        if self.__class__.__name__ == "UnitreeB2RampWalkEnvCfg":
            self.disable_zero_weight_rewards()

        # Warm-start note (matches every sibling terrain-only fork's own
        # comment, e.g. crawl/rear_stand): WALK's own obs/action shapes are
        # untouched by this file (no height_scan added, unlike vision_env_cfg's
        # own terrain-only fork) -- a WALK checkpoint should warm-start this
        # run directly via --resume --checkpoint, same recipe as everywhere
        # else in this directory.
