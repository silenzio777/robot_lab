# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeB2RoughEnvCfg


def crawl_base_height_l2(
    env,
    command_name: str,
    command_threshold: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """`mdp.base_height_l2`, gated to fire ONLY while actively crawling
    (|command| > threshold) -- the exact inverted form of the gate
    `feet_height_body` already uses in mdp/rewards.py (`> threshold` there,
    vs `stand_still_without_cmd`'s own `< threshold`), reused here rather
    than invented fresh.

    2026-08-22 CRAWLING redo (base+train design, see train_research/
    CRAWLING_REDO_TASK.md and TRAINING_STATE.md same date): stock
    `base_height_l2` has no command-gate param at all, so this repo-local
    wrapper exists specifically to add one -- editing the SHARED mdp.
    base_height_l2 itself was rejected to avoid touching rough/walk/
    leg_lift's own unrelated use of that term. Without this gate, crawl
    stayed in its low crouch target even at cmd=0 (idle), conflicting with
    the shared-anchor requirement that every skill's idle pose matches
    `stage_a_standing`'s tall stance. Un-gated, `base_height_l2` fires
    every timestep regardless of command -- gating it to "moving only"
    and restoring `stand_still_without_cmd` (see this file's own
    __post_init__) to the stock value it had before 2026-08-02's redesign
    cleanly separates the two regimes instead of fighting over one term."""
    reward = mdp.base_height_l2(env, target_height=target_height, asset_cfg=asset_cfg, sensor_cfg=sensor_cfg)
    reward = reward * (torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > command_threshold)
    return reward


@configclass
class UnitreeB2CrawlRoughEnvCfg(UnitreeB2RoughEnvCfg):
    """Crawl variant of UnitreeB2RoughEnvCfg -- same task/observations/actions, only the
    reward shaping changes, so a checkpoint trained under UnitreeB2RoughEnvCfg (e.g.
    model_5000.pt from unitree_b2_rough) can warm-start this run directly: same obs dim,
    action dim, and network architecture, just a different reward landscape to fine-tune
    into. See `logs/rsl_rl/unitree_b2_crawl/`'s own seed-run directory for how the
    warm-start checkpoint gets placed there (train_research/robot_lab/README.md).

    Only `base_height_l2` and `feet_height_body` change from the walking recipe:

    - `base_height_l2` (weight 0 -> -8.0, target 0.53 -> 0.35): the walking recipe
      deliberately leaves this at weight=0 (matches upstream fan-ziqi/robot_lab's own
      recipe) -- there's no reward term anywhere pushing the robot toward a specific
      height, so nothing here fights against staying low. -8.0/0.35 are starting guesses,
      not calibrated against real training curves yet -- watch `base_height_l2` in
      TensorBoard and retune the weight if the crouch looks too loose/too stiff.

      IMPORTANT, and the reason target_height is 0.35 and not lower: this exact reward
      shape (an `upward` orientation-only term with height left unrewarded) already
      produced a real reward-hacking collapse once in this project's history -- see
      TRAIN.md's "Reward hack: блин" entry (a *different*, now-abandoned training branch,
      unitree_rl_lab, but the same underlying reward-shape hazard: `upward =
      (1-projected_gravity_z)^2` is maximal for ANY level torso, standing OR lying flat on
      the ground, since it only measures orientation, never height -- confirmed by reading
      robot_lab's own `mdp.rewards.upward`, an identical formula). That failure needed a
      hard height-floor *termination* to fix (this project's current robot_lab-based
      TerminationsCfg has no such floor -- only time_out/terrain_out_of_bounds/
      illegal_contact). `base_height_l2` is a two-sided squared-error penalty around
      its target, though, not a one-sided floor -- as long as target_height is picked
      meaningfully above a collapsed/prone pose's height (per TRAIN.md, "блин" settled
      around ~0.2m), the SAME term that pulls the robot down from standing (0.53m) also
      pulls it back up if it ever sags toward lying flat, without needing a new
      termination. Do not lower target_height further without also reintroducing some
      kind of floor guard -- getting too close to the "лежит" height starts recreating
      the exact conditions that exploit produced.
    - `feet_height_body` (weight -5.0 -> 0): the walking recipe rewards feet sitting
      ~0.4m below the body (`target_height=-0.4`), calibrated for a 0.53m standing
      height -- physically impossible to satisfy at a 0.35m crouch (would put feet
      ~0.05m *below* the ground plane), so left as-is it would fight the new
      base_height_l2 target from the opposite direction. Disabled here rather than
      guessing a new target without real crouch-kinematics data; revisit once training
      shows what a natural low stance's own foot/body geometry actually looks like.

    `upward` is deliberately left untouched (still 3.0) -- it's a pure torso-orientation
    term (per its own formula, does not depend on height at all), still useful for
    discouraging toppling sideways, and per the analysis above it does not by itself
    cause or prevent the height issue -- that's base_height_l2's job.

    2026-08-02, round 2 (found from a real trained checkpoint, model_10300.pt): the
    crouch was correct under ANY nonzero command but reverted to the full standing
    height whenever the commanded velocity was near zero. Root cause -- TWO reward
    terms inherited unexamined from the walking recipe both anchor "standing still" to
    `asset.data.default_joint_pos` (the robot ASSET's own init_state pose, i.e. the tall
    standing stance, completely unrelated to this crawl variant's own target), and both
    are gated to fire specifically in the near-zero-command regime
    (`command_threshold=0.1`, see mdp/rewards.py's own `stand_still_without_cmd`/
    `joint_pos_penalty`):
    - `stand_still_without_cmd` (weight -2.0): active ONLY near zero command, a flat L1
      pull straight back to the tall standing joint angles. Disabled entirely here
      (weight 0) -- base_height_l2 (active at every timestep, moving or not) is the
      right, and now only, authority on height; there's no crawl-specific "ideal
      standstill pose" to substitute in its place, so removing the term outright (not
      retargeting it) is the correct fix, not a placeholder.
    - `joint_pos_penalty` (weight -1.0, always active): amplifies its own already-active
      pull toward the same tall default_joint_pos by `stand_still_scale=5.0x`
      specifically when nearly stationary. Left the term itself active (still useful in
      general -- discourages the crouch from being an unnecessarily wild/jerky joint
      configuration even while moving) but dropped `stand_still_scale` to 1.0 (no
      standstill-specific amplification) so it no longer singles out the at-rest case
      for extra pressure back toward standing tall.

    Confirmed this isn't a robot_stand (deployment) bug before touching training at all
    -- `B2LocomotionPolicy.default_leg_pos` (the constant its raw network output gets
    added to) is loaded once at construction and never varies with the live command
    value; nothing on the deployment side substitutes a different anchor pose at
    cmd_vel=0. The fix belongs entirely here, in reward shaping.

    2026-08-22 CRAWLING redo (base+train design, owner-confirmed decision -- see
    train_research/CRAWLING_REDO_TASK.md/TRAINING_STATE.md same date): the 2026-08-02
    reasoning above ("no crawl-specific ideal standstill pose to substitute in
    [stand_still_without_cmd's] place") no longer holds -- `confirmed_skills/
    stage_a_standing` (leg_lift Stage A's confirmed tall-standing checkpoint) now IS
    that pose, shared across every redo'd skill (crawling/rear_stand/walk) so
    switching between them on real hardware doesn't jerk. Owner ruled the
    always-low-crouch-even-at-idle behavior a BUG (same class of accidental stock-
    idle-package zeroing that cost leg_lift 9 failed versions), not a deliberate
    feature (e.g. ground clearance held even while standing under an obstacle) --
    reversible if that call turns out wrong.

    Fix: `base_height_l2` gated to fire ONLY while actively crawling
    (`crawl_base_height_l2`, this module -- reuses `feet_height_body`'s own existing
    `> command_threshold` gate pattern instead of inventing a new one), freeing
    `stand_still_without_cmd` to be restored to its stock weight for the idle
    regime. The two terms now own disjoint regimes instead of one unconditionally
    overpowering the other. Resumes from `stage_a_standing`'s own checkpoint
    (--resume --checkpoint, NEW seed directory) rather than the original walking
    donor -- see the launch command in TRAINING_STATE.md. Does NOT touch
    `confirmed_skills/crawling`'s own bench-button checkpoint/onnx.
    """

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards -- see class docstring for the full reasoning
        #
        # 2026-08-22 CRAWLING redo (base+train design, shared-idle-anchor
        # requirement -- see crawl_base_height_l2's own docstring above for
        # the full rationale): base_height_l2 now gated to fire ONLY while
        # actively crawling (command_threshold=0.1, same convention as
        # feet_height_body/stand_still_without_cmd elsewhere in this repo),
        # NOT every timestep regardless of command as before. This is what
        # makes restoring stand_still_without_cmd to its stock weight (below)
        # safe -- the two terms now own disjoint regimes (moving vs idle)
        # instead of base_height_l2 unconditionally overpowering it.
        self.rewards.base_height_l2 = RewTerm(
            func=crawl_base_height_l2,
            weight=-8.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "target_height": 0.35,
            },
        )
        self.rewards.feet_height_body.weight = 0
        # 0 -> -2.0 (2026-08-22 redo): restored to the stock value this term
        # had before 2026-08-02's redesign zeroed it. That redesign's own
        # reasoning ("no crawl-specific ideal standstill pose to substitute
        # in its place") no longer holds -- stage_a_standing now IS that
        # pose, shared across every skill for FSM-transition smoothness on
        # real hardware. See CRAWLING_REDO_TASK.md/TRAINING_STATE.md
        # 2026-08-22 for the full owner-confirmed decision (bug, not a
        # deliberate low-crouch-always feature).
        self.rewards.stand_still_without_cmd.weight = -2.0
        self.rewards.joint_pos_penalty.params["stand_still_scale"] = 1.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeB2CrawlRoughEnvCfg":
            self.disable_zero_weight_rewards()
