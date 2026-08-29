# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Handstand walk for B2 -- front-leg-only support with a torso pivoted vertical,
rear end up ("real" handstand, tail-up, as opposed to `rear_stand_env_cfg.py`'s
nose-up rear-leg stand -- the two are mirror images, not the same skill).

Owner's request 2026-08-30, after reading `extreme-parkour` (CMU/ICRA2024,
`~/base/DISTILLED/2026-08-30_extreme-parkour-paper-full-read.md`): port their
BLIND (proprioception-only, no depth camera -- the paper's own handstand-walk
policy trains with no vision at all and still crosses stairs backward blind)
front-leg handstand-walk and ramp-traversal skills to B2, skipping the vision
side entirely (our RealSense is not even physically confirmed present on this
B2, see `~/base/ROBOTS/B2/HARDWARE.md` #3 -- not a blocker for a blind skill).

**CORRECTION (owner, 2026-08-30, same session -- "почему опять robot_lab то?!
У нас ничего особо с их конфигами не получилось"):** the orientation reward
below was FIRST ported from robot_lab's stock `config/others/
unitree_a1_handstand/` reference -- checked its git history directly
(`git log -- .../unitree_a1_handstand/`) and found ONLY structural commits
("move dir", "add CusRL support") by fan-ziqi, zero evidence anyone ever
trained it, no README mention anywhere in this repo. A fair, now-confirmed
concern -- that file is an untested template, not a validated recipe, and
this lab's own track record (rear_stand's 6+ postmortem saga, JUMP v10's
still-open thigh-exploit) is exactly why "it's in the repo" isn't evidence
it works. Re-checked directly against `extreme-parkour`'s own PUBLIC CODE
(not just the paper) for its actual handstand mechanism -- full-tree grep of
`github.com/chengxuxin/extreme-parkour` for "hand"/"stand" (case-insensitive)
matches NOTHING in any .py file; the string "handstand" does not appear
anywhere in their released code. The capability is real (paper section 4.2.3,
videos) but the concrete reward implementation was never open-sourced --
only the general PRINCIPLE is published (`r_stylized`, eq.4: inner product
between the body's world-frame forward axis and a switchable target
direction). **Orientation reward below is now built directly from that
published formula** (`stylized_forward_tracking`, this file, module level) --
grounded in the peer-reviewed paper's own math, not in robot_lab's
unvalidated stand-in. The feet-height/air-time terms (still sourced from the
robot_lab reference) are kept as a secondary, structurally uncontroversial
shaping signal (matches the paper's own qualitative description, "kicks
upward with rear legs... keeps them in neutral pose") -- not the part that
was in question.

Structural port from `config/others/unitree_a1_handstand/` (`handstand_type=
"back"` = rear feet airborne = front-leg support = literal handstand) is
still the base for the SCAFFOLDING (which bodies count as support vs
airborne, contact/termination wiring) -- that part is robot-mechanics
bookkeeping, not a trained-behavior claim, and needed the same THREE B2-
specific corrections regardless of where the orientation term itself came
from:

1. **foot_link_name divergence (the real risk).** A1's URDF keeps a distinct
   foot body separate from the calf; B2's URDF merges the foot into the calf
   via a fixed joint (confirmed in `rough_env_cfg.py`'s own foot_link_name
   comment: `merge_fixed_joints=True` absorbs it, `foot_link_name=".*_calf"`).
   The A1 reference's own `illegal_contact` termination exempts foot_link_name
   wholesale (`^(?!.*_foot).*`) -- harmless on A1 since that only exempts the
   foot TIP. Copied verbatim onto B2 it would exempt the ENTIRE rear calf
   segment (foot_link_name==".*_calf" covers both front AND rear calves) from
   the hard safety termination -- silently reopening the exact "calf-sitting"
   exploit `rear_stand_env_cfg.py`'s own v6.1-v6.5 saga fought for days to
   close (a leg resting on its calf instead of genuinely lifting scores
   identically to a real lift unless something explicitly polices it). Fixed
   below: illegal_contact excludes ONLY the front calves (the real support
   surface), rear calf contact terminates like any other illegitimate contact.
2. **Geometry-grounded target height, not A1's borrowed 0.5m.** B2's front/rear
   hip offsets from base_link are IDENTICAL by construction (URDF: FR_hip
   origin x=+0.3285, RR_hip origin x=-0.3285 -- confirmed by direct grep,
   2026-08-30) -- so the same reasoning `rear_stand_env_cfg.py`'s
   FRONT_FEET_HEIGHT_TARGET already used (STAND_HEIGHT_TARGET 0.62m + 0.3285m
   hip offset =~0.95m theoretical, target set to 0.85m to leave functional
   knee bend rather than demand a mechanical singularity) applies to the
   MIRROR case by the same symmetry -- reused directly below as
   REAR_FEET_HEIGHT_TARGET, not re-derived from scratch or borrowed from a
   12kg A1's own unrelated calibration.
3. **B2 is ~6x heavier than A1** (73.55kg vs ~12kg) -- rear_stand's own
   multi-night saga (v6.1-v6.5) already proved a bare orientation+feet-height
   reward pair is NOT enough anchor for a robot this heavy to find a genuine
   stance instead of a "torso-tilt substitutes for real extension" local
   optimum; the A1 reference disables base_height_l2 entirely (weight=0).
   Kept DISABLED here too for this v1 (closest-possible port, minimize
   simultaneous changes) but flagged explicitly: if the first bench/training
   pass shows the same plateau signature rear_stand hit (orientation solves,
   feet-height stalls mid-climb, visible torso-tilt substitute), the fix is
   already known and cheap -- add a root-height anchor the same way
   rear_stand_height did, don't re-diagnose from zero.

**STAGING, matching the owner's own already-proven-necessary rule for the
mirror skill** ("Нужно делать по частям! Сначала встаем на задние" --
rear_stand v6, after v3-v5's combined stand+walk+turn attempt trained walking
to literally near-zero: walk_tracking 0.16/turn_tracking 0.10/rear_feet_
contact 0.18 in the final metrics, THREE FULL RUNS never producing real
bipedal walking despite being "in" the reward economy the whole time).

This file is Stage A: **handstand BALANCE only**, no forward/turn command
active (velocity-tracking terms zeroed below, unlike the A1 reference which
leaves them live) -- deliberately narrower than what the owner asked for
("handstand-ходьбу") because rear_stand's own hard-won lesson is that
walking-while-balanced-on-2-legs is a qualitatively harder problem than
balancing, and combining both from iteration 0 has ALREADY been tried once on
this exact class of skill and failed outright. Once Stage A holds solid on
the bench (a real, still handstand, not a wobble/topple), Stage B re-enables
velocity tracking the same way rear_stand's own v2/v3 staged it in -- warm-
started FROM this checkpoint, not trained from scratch, same technique
`RearStandStageACommand`'s own docstring documents for stage_a_standing.

Ramp traversal (the OTHER half of the owner's request, up to 37 degrees) is a
SEPARATE, unrelated skill -- ordinary quadruped walking on steep terrain, not
a leg-support-topology change -- see `ramp_walk_env_cfg.py` in this same
directory, built on WALK's own already-live reward economy instead."""

import torch

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.config.others.unitree_a1_handstand.env import (
    rewards as handstand_rewards,
)

from .rough_env_cfg import UnitreeB2RoughEnvCfg

# Reused verbatim from rear_stand_env_cfg.py's own FRONT_FEET_HEIGHT_TARGET --
# see module docstring point 2 for the full symmetry argument (identical
# 0.3285m front/rear hip offsets, confirmed against b2_description.urdf
# directly, 2026-08-30). Renamed here since THIS file's airborne feet are the
# REAR ones (front-leg support), not the front ones rear_stand tracks.
REAR_FEET_HEIGHT_TARGET = 0.85
REAR_FEET_HEIGHT_IDLE = 0.15  # same idle (4-leg stance) baseline rear_stand's own constant used
REAR_FEET_HEIGHT_SIGMA = 0.25  # matches A1 reference's own std=sqrt(0.25) -> sigma=0.25 (this file's exp(-err/sigma) convention)

# extreme-parkour's own target direction for handstand ([0,0,-1] -- see
# stylized_forward_tracking's own docstring): nose points straight DOWN in
# world frame once genuinely vertical, tail-up. Consistent with the paper's
# own section 4.2.3 description ("bends forward and shifts weight onto front
# legs... kicks upward with rear legs") and independently checked against
# this file's own REAR_FEET_HEIGHT_TARGET geometry (same vertical posture,
# different observable -- foot height vs. body-axis direction).
HANDSTAND_TARGET_DIR = (0.0, 0.0, -1.0)


def stylized_forward_tracking(env, target_dir: tuple[float, float, float], asset_cfg=None) -> torch.Tensor:
    """extreme-parkour's own r_stylized (arXiv 2309.14341 eq.4, CMU/ICRA2024) --
    read directly from the PAPER's math, not from any released code (confirmed
    2026-08-30: the string "handstand" does not appear anywhere in
    github.com/chengxuxin/extreme-parkour's actual source -- see this file's
    own module docstring for the full correction/reasoning). Their formula:

        r_stylized = W * [0.5*<v_fwd, c> + 0.5]^2

    `v_fwd` is the body's own forward (+X) axis in WORLD frame, `c` a fixed
    target direction, `W` a binary on/off switch the paper uses to toggle
    between skills at deployment (walk vs handstand) -- W is just this
    RewTerm's own `weight` here, no separate implementation needed.

    World-frame forward-axis formula matches this directory's own already-
    verified convention EXACTLY: `rear_stand_env_cfg.py`'s `_fwd_axis_z`
    returns `2*(x*z - w*y)` for the Z-component alone; the full 3-vector
    (needed here since `target_dir` isn't restricted to world-Z) is the
    standard quaternion-rotation first row, X/Y components added consistently
    with that same verified Z formula, not re-derived independently.

    At target_dir=(0,0,-1) (this file's HANDSTAND_TARGET_DIR): fwd_z=-1
    (nose fully down, genuine handstand) -> reward 1.0; fwd_z=+1 (nose up,
    e.g. rear_stand's OWN target -- wrong direction for THIS skill) -> reward
    0.0; fwd_z=0 (ordinary flat standing) -> reward 0.25, a smooth partial
    gradient from any starting orientation, matching the paper's own claim
    for why this formula (vs. a narrow exp-kernel) works from a cold start."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    q = asset.data.root_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    fwd_x = 1.0 - 2.0 * (y * y + z * z)
    fwd_y = 2.0 * (x * y + w * z)
    fwd_z = 2.0 * (x * z - w * y)  # matches rear_stand_env_cfg.py's own _fwd_axis_z exactly
    tx, ty, tz = target_dir
    dot = fwd_x * tx + fwd_y * ty + fwd_z * tz
    return torch.square(0.5 * dot + 0.5)


@configclass
class UnitreeB2HandstandEnvCfg(UnitreeB2RoughEnvCfg):
    """Stage A: static handstand balance, no walk/turn command (see module
    docstring's staging section). New reward terms added dynamically in
    __post_init__ (same convention every sibling file in this directory
    uses -- e.g. walk_env_cfg.py's own self.rewards.walk_pair_match = RewTerm(...)
    -- not a new RewardsCfg subclass)."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0  # matches A1 reference

        # ------------------------------Rewards: 4 new handstand terms, imported not duplicated------------------------------
        # Functions are robot-agnostic (take asset_cfg/sensor_cfg params, no
        # A1-specific body names baked in -- confirmed reading env/rewards.py
        # directly) -- reused from the stock reference, not copy-pasted.
        self.rewards.handstand_feet_height_exp = RewTerm(
            func=handstand_rewards.handstand_feet_height_exp,
            weight=0.0,
            params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.0, "std": REAR_FEET_HEIGHT_SIGMA**0.5},
        )
        self.rewards.handstand_feet_on_air = RewTerm(
            func=handstand_rewards.handstand_feet_on_air,
            weight=0.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="")},
        )
        self.rewards.handstand_feet_air_time = RewTerm(
            func=handstand_rewards.handstand_feet_air_time,
            weight=0.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=""), "threshold": 5.0},
        )
        # NOT handstand_rewards.handstand_orientation_l2 (robot_lab's own,
        # unvalidated) -- see module docstring's correction. Uses this file's
        # own stylized_forward_tracking, extreme-parkour's published r_stylized.
        self.rewards.stylized_forward_tracking = RewTerm(
            func=stylized_forward_tracking,
            weight=0.0,
            params={"target_dir": HANDSTAND_TARGET_DIR},
        )

        # ------------------------------Rewards: strip the normal quadruped anchors------------------------------
        # Same zeroing list as A1 reference -- these all pull toward a FLAT,
        # four-legged stance, which is the opposite of this task.
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = 0
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0  # see module docstring point 3 -- flagged, not silently trusted
        self.rewards.stand_still_without_cmd.weight = 0
        self.rewards.feet_air_time.weight = 0
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_slide.weight = 0
        self.rewards.contact_forces.weight = 0

        # ------------------------------Rewards: Stage A -- no velocity command------------------------------
        # UNLIKE the A1 reference (which leaves track_lin_vel_xy_exp/track_ang_vel_z_exp
        # live at weight 3.0/1.5, i.e. attempts walk+handstand together from
        # iteration 0) -- zeroed here per this file's own staging argument
        # (module docstring: rear_stand's v3-v5 already tried "everything at
        # once" on the mirror skill and it trained walking to near-zero).
        self.rewards.track_lin_vel_xy_exp.weight = 0
        self.rewards.track_ang_vel_z_exp.weight = 0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False

        # ------------------------------Rewards: regularizers, A1 reference's own values------------------------------
        # Kept as-is (not B2-mass-recalibrated) for this v1 -- flag to re-check
        # if torque/action-rate trends look wrong on the first real run, same
        # "first guess, calibrate from what training produces" convention
        # every other file in this directory already uses for its own first pass.
        self.rewards.joint_torques_l2.weight = -1e-3
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-6
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -2e-4
        self.rewards.action_rate_l2.weight = -0.05

        # ------------------------------Rewards: HandStand (type="back" -- front-leg support, literal handstand)------------------------------
        # air_foot_name uses B2's OWN foot_link_name convention (".*_calf",
        # see module docstring point 1) -- NOT A1's ".*_foot", which does not
        # exist as a separate body on B2.
        air_foot_name = "R.*_calf"
        # POSITIVE weight -- stylized_forward_tracking returns a reward to
        # MAXIMIZE (1.0=correct orientation), unlike robot_lab's own
        # handstand_orientation_l2 (an L2 penalty, minimized at 0, hence its
        # negative weight) -- not the same sign convention, don't copy -1.0.
        # Magnitude 10.0 matches handstand_feet_height_exp's own weight below
        # (this file's other primary orientation-adjacent anchor) -- both are
        # THE dominant terms this task should optimize, first guess on the
        # exact ratio between them, calibrate from what training produces.
        self.rewards.stylized_forward_tracking.weight = 10.0
        self.rewards.handstand_feet_height_exp.params["target_height"] = REAR_FEET_HEIGHT_TARGET
        self.rewards.handstand_feet_height_exp.weight = 10.0
        self.rewards.handstand_feet_height_exp.params["asset_cfg"].body_names = [air_foot_name]
        self.rewards.handstand_feet_on_air.weight = 5.0
        self.rewards.handstand_feet_on_air.params["sensor_cfg"].body_names = [air_foot_name]
        self.rewards.handstand_feet_air_time.weight = 5.0
        self.rewards.handstand_feet_air_time.params["sensor_cfg"].body_names = [air_foot_name]

        # Soft early-warning penalty on any REAR ground contact (thigh OR calf --
        # module docstring point 1, the calf-sitting exploit class) BEFORE the
        # hard termination below fires -- same two-tier pattern (soft reward +
        # hard termination) rear_stand/jump_v10 both already use.
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["R.*_thigh", "R.*_calf"]

        if self.__class__.__name__ == "UnitreeB2HandstandEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # rough_env_cfg.py sets self.terminations.illegal_contact = None (disabled
        # for ordinary locomotion, where transient non-foot contact is tolerable)
        # -- FIX (train, 2026-08-30, found at launch: base's original line assumed
        # this was still a live DoneTerm and mutated .params on None, an
        # AttributeError at env construction, not just a comment-vs-code drift --
        # py_compile can't catch this, it's a runtime attribute access): re-enable
        # the STOCK illegal_contact termination directly (same mechanism jump_v10
        # re-enables via its own grace-period wrapper, see jump_v10_env_cfg.py's
        # own comment on this exact line) rather than assuming it already exists.
        # Excludes ONLY the front calves (legitimate support surface) -- NOT A1's
        # foot_link_name-wide exemption, see module docstring point 1. Rear calf
        # contact (dragging/resting instead of genuinely lifting) terminates the
        # episode like base/hip/thigh contact would. Stock threshold=1.0N kept
        # (velocity_env_cfg.py's own default) -- this term flags ANY non-trivial
        # contact on a monitored body, not a mass-scaled force limit, so B2's much
        # higher weight than A1 doesn't change what the right threshold is.
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["^(?!.*F.*_calf).*"]),
                "threshold": 1.0,
            },
        )

        # ------------------------------Curriculums------------------------------
        self.curriculum.command_levels = None
