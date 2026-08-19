# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Single-leg lift ("трипод") for B2 -- a deliberately simple test task (user, 2026-08-09:
"он простой, тестовый, я хочу понять что можно делать через RL а что придётся через IL").

Setup: the robot stands stable on all four legs. A held direction command lifts exactly
ONE leg and holds the other three as a tripod support, torso free to lean for balance
(not held flat -- that's the user's own explicit spec, not an oversight):
  - forward -> front-left  (FL)
  - right   -> front-right (FR)
  - back    -> rear-right  (RR)
  - left    -> rear-left   (RL)

Structurally closest to rear_stand's OWN v1/v2 (a held ramped signal drives a static
target pose, not a periodic gait -- see feet_air_time/feet_gait staying at their rough
defaults of 0, same "not a gait" reasoning as jump_env_cfg's own docstring) crossed with
jump's direction-vector command idiom (a unit vector picks WHICH of several discrete
things happens, see JumpPulseCommand). Command stays the same 3 slots every other
non-vision B2 variant uses ([signal, dir_x, dir_y]) -- NOT jump's 5-slot rear-stand-v4
kind of growth -- so a plain 45-obs checkpoint (e.g. the standard rough walking run) can
warm-start this one if desired; nothing here needs a wider observation.
"""

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeB2RoughEnvCfg

# Command timing -- first guess, calibrate from what training produces (same idiom as
# every other first-guess constant in this repo, e.g. rear_stand's own RISE_DURATION).
# Much faster than rear_stand's whole-body rise (2.0s): a single leg lifting is a far
# smaller, lighter motion than rearing the whole robot up.
IDLE_TIME_RANGE = (1.5, 3.0)
RISE_DURATION = 0.6
HOLD_TIME_RANGE = (1.5, 3.0)
DESCEND_DURATION = 0.6
# 0.12 -> 0.15 (2026-08-12, user: the fold should be pronounced -- "вверх...
# и сильнее" -- and with LIFT_XY_TOLERANCE now pinning the foot under the hip,
# height can only come from a genuine upward fold).
LIFT_HEIGHT_TARGET = 0.15  # m, foot clearance above the support tripod
TRACKING_SIGMA = 0.01  # m^2, sharp -- clearance error is naturally small-scale (meters)
# Own constant for leg_lift_selected_height's proximity_gate (2026-08-13 bench-monitor
# autonomy fix, it11255/25000 of the gated-height v3 run): reusing TRACKING_SIGMA there
# gave gate~0.6 at the observed steady-state excess (~0.06-0.10m, 3 straight ~1200-it
# buckets sitting in the same 13-18cm band with reward/vloss no longer improving) --
# not enough gradient to keep closing the gap. A SEPARATE constant lets the gate get
# sharper WITHOUT also tightening height_kernel's own clearance-tracking tolerance
# (an unrelated, currently-working part of the same reward term that happens to share
# the old constant only by the docstring's own "same scale" convenience, not a real
# coupling requirement). At excess=0 this changes nothing (exp(0)=1 regardless of
# sigma) -- only genuinely swept-back feet get pushed harder.
LIFT_XY_GATE_SIGMA = 0.004  # m^2, ~2.5x sharper than TRACKING_SIGMA

# Direction unit vectors, FL/FR/RR/RL order -- matches _selected_leg_mask's own index
# convention throughout this file. forward=FL, right=FR, back=RR, left=RL (user's own
# spec, confirmed 2026-08-09 after the written spec's "left" line had a copy-paste typo
# repeating FR).
LEG_DIRECTIONS = ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0))

# Explicit, ORDERED (FL,FR,RR,RL) name lists -- every per-leg-shaped reward function
# below relies on this exact order matching LEG_DIRECTIONS/_selected_leg_mask's index
# convention. SceneEntityCfg preserves an explicit list's order (unlike a regex, which
# resolves in the articulation's own body-list order) -- same reliance every other
# per-leg-ordered array in this codebase already makes (e.g. b2_policy.py's
# leg_joint_names-driven qpos/qvel/ctrl address arrays on the bench side).
FOOT_BODY_NAMES = ["FL_calf", "FR_calf", "RR_calf", "RL_calf"]
HIP_BODY_NAMES = ["FL_hip", "FR_hip", "RR_hip", "RL_hip"]
# Lifted foot must stay horizontally near its own hip -- band before the penalty
# bites. 0.2 -> 0.08 (2026-08-12, bench on 24999 WITH the fixed driver: every
# direction now picks the right leg, but the "lift" is still a 10-15cm BACKWARD
# sweep -- which sits entirely INSIDE the old 0.2 band, so the penalty never
# fired once. 0.08 makes any sweep bite immediately; gaining clearance then
# mechanically requires folding thigh+calf upward, which is the user's explicit
# spec: "нужно не НАЗАД а ВВЕРХ... повернуть эти суставы в обратную сторону").
LIFT_XY_TOLERANCE = 0.08
# Base height anchor while a lift is commanded -- rough's own standing target.
LIFT_BASE_HEIGHT_TARGET = 0.53
LEG_JOINT_NAMES_ORDERED = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]  # fmt: skip


class LegLiftCommand(CommandTerm):
    """Cycle: idle (four legs) -> rise -> hold (one leg lifted) -> descend -> resample,
    same clock trick as jump's JumpPulseCommand / rear_stand's RearStandCommand:
    _resample_command overwrites the base class's time_left with the full cycle length.
    Direction is picked once per cycle (which leg lifts) and held zero until rise starts,
    same "idle reads as an honest all-zero command" idiom used everywhere else in this
    repo (jump's own window-gated direction, rear_stand's ramp-gated walk/turn)."""

    cfg: "LegLiftCommandCfg"

    def __init__(self, cfg: "LegLiftCommandCfg", env) -> None:
        super().__init__(cfg, env)
        n = self.num_envs
        self._command = torch.zeros(n, 3, device=self.device)
        self.signal = torch.zeros(n, device=self.device)
        self.direction = torch.zeros(n, 2, device=self.device)
        self.idle_duration = torch.zeros(n, device=self.device)
        self.hold_duration = torch.zeros(n, device=self.device)
        self.cycle_duration = torch.zeros(n, device=self.device)
        self._directions = torch.tensor(cfg.directions, dtype=torch.float, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        self.metrics["lift_signal"] = self.signal.clone()

    def _resample_command(self, env_ids):
        idle = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.idle_time_range)
        hold = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.hold_time_range)
        self.idle_duration[env_ids] = idle
        self.hold_duration[env_ids] = hold
        self.cycle_duration[env_ids] = idle + self.cfg.rise_duration + hold + self.cfg.descend_duration
        self.time_left[env_ids] = self.cycle_duration[env_ids]
        dir_idx = torch.randint(0, self._directions.shape[0], (len(env_ids),), device=self.device)
        self.direction[env_ids] = self._directions[dir_idx]

    def _update_command(self):
        elapsed = self.cycle_duration - self.time_left
        rise_start = self.idle_duration
        hold_start = rise_start + self.cfg.rise_duration
        descend_start = hold_start + self.hold_duration
        rising = ((elapsed - rise_start) / self.cfg.rise_duration).clamp(0.0, 1.0)
        descending = ((elapsed - descend_start) / self.cfg.descend_duration).clamp(0.0, 1.0)
        self.signal = rising - descending  # 0 idle, ramps up, holds 1, ramps down
        self._command[:, 0] = self.signal
        active = (elapsed >= rise_start).float().unsqueeze(-1)
        self._command[:, 1:3] = self.direction * active


@configclass
class LegLiftCommandCfg(CommandTermCfg):
    class_type: type = LegLiftCommand
    resampling_time_range: tuple[float, float] = (4.0, 8.0)  # nominal; overwritten per cycle
    idle_time_range: tuple[float, float] = IDLE_TIME_RANGE
    rise_duration: float = RISE_DURATION
    hold_time_range: tuple[float, float] = HOLD_TIME_RANGE
    descend_duration: float = DESCEND_DURATION
    directions: tuple = LEG_DIRECTIONS
    debug_vis: bool = False


def _selected_leg_mask(env, command_name: str) -> torch.Tensor:
    """[num_envs, 4] one-hot (FL,FR,RR,RL order) of which leg the command currently
    selects, from the direction unit vector in command slots 1:3 (nearest by dot
    product -- exact match by construction, argmax just avoids a brittle float-equality
    check). Idle (direction=(0,0)) ties every dot product at 0 and argmax deterministically
    returns index 0 (FL) -- harmless, every reward using this mask also multiplies by
    lift_signal (0 at idle), so which leg nominally "wins" the idle tie never matters."""
    term = env.command_manager.get_term(command_name)
    direction = term.command[:, 1:3]
    dirs = torch.tensor(LEG_DIRECTIONS, device=direction.device, dtype=direction.dtype)
    idx = torch.argmax(direction @ dirs.T, dim=1)
    mask = torch.zeros(direction.shape[0], 4, device=direction.device, dtype=direction.dtype)
    mask.scatter_(1, idx.unsqueeze(-1), 1.0)
    return mask


def leg_lift_selected_height(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    hip_cfg: SceneEntityCfg,
    target_lift: float,
    xy_tolerance: float,
) -> torch.Tensor:
    """Track the COMMANDED leg's foot clearance above the OTHER THREE's own average
    height -- self-calibrating (relative to the live support tripod, not an absolute
    world-Z guess), so it needs no ground-offset constant tuned by hand the way an
    absolute-height anchor would. Target is target_lift*signal: 0 at idle (all four
    flat), target_lift once fully commanded -- same signal-scaled-target idiom as
    jump_idle_height / rear_stand_orientation_tracking.

    GATED by horizontal proximity to the hip (added 2026-08-13, bench verdict on
    30700/30900, TWO checkpoints AFTER the 2026-08-13-night support_pose loosening:
    "поднимает ноги НАЗАД" -- STILL backward, unchanged). Root cause of why the
    previous fix (loosening leg_lift_support_pose, and before that adding
    leg_lift_foot_horizontal as an independent side-penalty) didn't work: height and
    horizontal-excess were two SEPARATE additive terms competing on the SAME reward
    sum, so the policy could keep paying foot_horizontal's penalty AS LONG AS the
    height payout still won on net -- and back-calculating from the logged
    foot_horizontal component (~-0.19 episode-average at lift_signal~0.58) puts the
    actual sweep at roughly 0.30m past the hip, nowhere near the 0.08m tolerance.
    Loosening support_pose only freed UP BUDGET for the policy to keep affording that
    exact trade, it never made the trade itself unprofitable.

    Fix: multiply the height kernel by a horizontal-proximity gate instead of merely
    penalizing distance alongside it -- collecting ANY height credit now REQUIRES the
    foot to already be near the hip, so there is no longer a "pay the penalty, keep
    the height" trade to make; sweeping the leg back earns ~0 on BOTH terms at once.
    leg_lift_foot_horizontal is kept alongside as a second, independent deterrent
    (belt-and-suspenders), not relied on alone this time."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # [N,4], FL,FR,RR,RL order
    mask = _selected_leg_mask(env, command_name)
    selected_z = (foot_z * mask).sum(dim=1)
    support_z = (foot_z * (1.0 - mask)).sum(dim=1) / 3.0
    clearance = selected_z - support_z
    target = target_lift * term.signal
    height_kernel = torch.exp(-torch.square(clearance - target) / TRACKING_SIGMA)

    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, 0:2]
    hip_xy = asset.data.body_pos_w[:, hip_cfg.body_ids, 0:2]
    dist = torch.linalg.norm(foot_xy - hip_xy, dim=-1)  # [N,4]
    selected_dist = (dist * mask).sum(dim=1)
    excess = (selected_dist - xy_tolerance).clamp(min=0.0)
    # LIFT_XY_GATE_SIGMA (own constant, see its own comment above -- 2026-08-13,
    # sharpened from the height_kernel's shared TRACKING_SIGMA once that proved too
    # forgiving at the observed steady-state excess): at a genuine near-hip fold
    # (<0.08m, excess=0) this is exactly 1.0 regardless of sigma and never discounts
    # an honest lift; it only sharpens the falloff for genuinely swept-back feet.
    proximity_gate = torch.exp(-torch.square(excess) / LIFT_XY_GATE_SIGMA)

    return height_kernel * proximity_gate


def leg_lift_support_contact(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the THREE support feet losing contact while a lift is actually
    commanded -- the tripod must stay a genuine tripod, not drift into a fourth
    free-floating configuration nothing else prices. Gated by lift_signal (no penalty
    during idle/ramp-in) since only a full lift actually demands 3-point support."""
    term = env.command_manager.get_term(command_name)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = (torch.linalg.norm(forces, dim=-1) > 1.0).float()  # [N,4]
    mask = _selected_leg_mask(env, command_name)
    missing = (1.0 - in_contact) * (1.0 - mask)
    return torch.sum(missing, dim=1) * term.signal


def leg_lift_foot_horizontal(
    env, command_name: str, foot_cfg: SceneEntityCfg, hip_cfg: SceneEntityCfg, tolerance: float
) -> torch.Tensor:
    """Penalty on the LIFTED foot's horizontal distance from its own hip beyond a
    tolerance band -- "поднять" means fold the leg UP under the hip, not sweep it
    away. Added 2026-08-11 (bench verdict on model_7999: the one leg that responds
    at all (RR) pulls BACKWARD instead of up -- leg_lift_selected_height prices
    only the VERTICAL clearance, so where the foot goes horizontally was a free
    variable, and swinging the leg back happens to be the physically cheapest way
    to gain a little clearance. The master free-variable lesson, instance N.)

    Distance measured hip-to-foot in the horizontal plane (both from live body
    poses, so it's orientation-robust), excess over the band squared, clamped
    (same bounded-worst-case discipline as every clamped term in jump_env_cfg),
    masked to the commanded leg, scaled by the lift signal."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    foot_xy = asset.data.body_pos_w[:, foot_cfg.body_ids, 0:2]  # [N,4,2] FL,FR,RR,RL
    hip_xy = asset.data.body_pos_w[:, hip_cfg.body_ids, 0:2]
    dist = torch.linalg.norm(foot_xy - hip_xy, dim=-1)  # [N,4]
    excess = (dist - tolerance).clamp(min=0.0, max=1.0)
    mask = _selected_leg_mask(env, command_name)
    selected_excess = (excess * mask).sum(dim=1)
    return torch.square(selected_excess) * term.signal


def leg_lift_base_height(env, command_name: str, target_height: float) -> torch.Tensor:
    """L2 base-height anchor active while a lift is commanded. Added 2026-08-11
    (bench verdict on model_7999: "сильно приседает" during the RR lift). Root
    cause: rough's own base_height_l2 is weight-0 in the parent config, and this
    file's support_pose penalty was the ONLY thing opposing a squat -- at -2.0 vs
    the height reward's +6.0, crouching (which lowers the CoM and makes the
    3-legged balance easier) was a cheap trade. Nothing anchored the base height
    at all -- the same free-variable hole jump_idle_height plugged for the jump
    task, plugged the same way here. Scaled by the lift signal: idle keeps its
    own anchors (upward + the retired-generic-terms replacement below)."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    err = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return err * term.signal


def leg_lift_base_still(env, command_name: str) -> torch.Tensor:
    """Penalty on horizontal base velocity + yaw rate, active in EVERY phase
    (added 2026-08-12, bench on 24999: the robot backpedals slowly during a
    commanded lift). Nothing in this task ever priced base translation --
    track_lin_vel/track_ang_vel are retired, stand_still_without_cmd is
    zeroed, and the whole task is defined as "stand in place and lift one
    leg" -- so pacing around was a free variable from day one (the same hole
    jump_idle_still plugged for the jump task; here the robot should never
    translate at all, so no phase gate)."""
    asset = env.scene["robot"]
    v_xy = torch.sum(torch.square(asset.data.root_lin_vel_w[:, 0:2]), dim=1)
    w_z = torch.square(asset.data.root_ang_vel_w[:, 2])
    return v_xy + w_z


def leg_lift_support_pose(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """L1 penalty on joint deviation from default, masked to the THREE support legs
    only -- the lifted leg's own joints must stay free to move (that's the whole
    trick), everything else should hold the plain stand exactly like idle does.

    Written in up front rather than discovered later on a bench test: this is the SAME
    class of bug as today's jump/rear_stand idle-pose regressions (see
    jump_idle_symmetry's own docstring in jump_env_cfg.py) -- the generic
    stand_still_without_cmd/joint_pos_penalty terms gate on command-NORM, all-or-
    nothing, so they'd fully deactivate for the entire duration of every lift (command
    is nonzero the moment a direction is picked) and leave the three support legs'
    pose completely unpriced exactly when it matters most. Zeroed those two generic
    terms in __post_init__ in favor of this one, correctly-masked, always-active
    anchor instead of leaving that gap for a checkpoint to drift into.

    v4 (2026-08-14, bench verdict on 20200): the exemption is now SCALED BY THE LIFT
    SIGNAL (`1 - mask*signal`) instead of binary (`1 - mask`). The binary version had
    exactly the hole the paragraph above tried to close, one leg over: at IDLE the
    command direction is (0,0), argmax resolves the tie to FL by default, so FL was
    permanently exempt from the anchor while standing -- and the checkpoint drifted
    into the bench-observed "FL stretched forward, body tilted ~10°, weird half-crouch"
    idle pose, plus the "кульбит" (the policy re-staged its front-leg stance per
    command, since stance re-arrangement was free). With signal-scaling, at signal=0
    ALL FOUR legs are anchored (symmetric rest is the only cheap pose), and the
    selected leg earns its freedom exactly in proportion to how far the lift has
    actually been commanded."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    diff = torch.abs(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    diff = diff.view(diff.shape[0], 4, 3)  # [N, leg(FL,FR,RR,RL), joint(hip,thigh,calf)]
    mask = _selected_leg_mask(env, command_name)
    exemption = mask * term.signal.unsqueeze(-1)
    support_diff = diff * (1.0 - exemption).unsqueeze(-1)
    return torch.sum(support_diff, dim=(1, 2))


class UnitreeB2LegLiftRoughEnvCfg(UnitreeB2RoughEnvCfg):
    """See module docstring -- a deliberately simple test skill, not a production one."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Flat ground -- same reasoning as jump_env_cfg's own: the trick is plenty hard
        # without rough terrain on top, revisit later if this graduates past "test".
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        # The lift command replaces the velocity command wholesale (same 3 slots, so
        # the observation layout -- and warm-start compatibility with any plain 45-obs
        # checkpoint -- is unchanged).
        self.commands.base_velocity = LegLiftCommandCfg()
        # command_levels_vel reads `.cfg.ranges` off the base_velocity term -- this
        # command has no ranges, same non-applicability as jump's own pulse command.
        self.curriculum.command_levels = None

        # v7 (2026-08-16): the v6 illegal_contact termination is REMOVED again --
        # back to v2's exact termination set (time_out/out_of_bounds only). It was
        # added to price the v5/v6 rear-command collapses, but v7 warm-starts from
        # v2's 24999, which never collapsed on the bench; keeping a termination the
        # base policy never trained under would be a gratuitous distribution shift
        # ("небольшими порциями, чтобы не запутать PPO" -- owner's staging order).

        # -- retire the velocity-tracking recipe, same as every other non-gait B2 variant
        self.rewards.track_lin_vel_xy_exp.weight = 0
        self.rewards.track_ang_vel_z_exp.weight = 0

        # Not a periodic gait (feet_air_time/feet_gait/feet_contact/feet_slide/
        # feet_stumble already default to 0 in rough_env_cfg, nothing to retire there).
        # feet_height/feet_height_body are generic ALL-feet swing-clearance terms
        # (walking-gait-shaped) -- retired in favor of leg_lift_selected_height below,
        # which is specific to exactly the one commanded leg.
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height_body.weight = 0

        # Single-leg lift is inherently an ASYMMETRIC motion (one leg moves, its
        # diagonal partner does not) -- joint_mirror (rough default -0.05, rewards
        # symmetry between diagonal FR<->RL / FL<->RR pairs) would directly fight the
        # entire point of this task.
        self.rewards.joint_mirror.weight = 0

        # See leg_lift_support_pose's own docstring for why these two generic terms
        # are retired here rather than left active alongside it.
        self.rewards.stand_still_without_cmd.weight = 0
        self.rewards.joint_pos_penalty.weight = 0

        # v4 (2026-08-14, bench verdict on checkpoint 20200 -- owner's live test +
        # frame review; TRAINING_STATE.md entry ~01:40 for the full story):
        # - flat_orientation_l2 0 -> -5.0: "ровный корпус ВСЕГДА" (owner's explicit
        #   rule). The bench-observed steady ~10° body tilt was nearly free: `upward`
        #   rewards the z-projection (cos(10°)≈0.985, costs ~1.5%), ang_vel_xy_l2
        #   prices rotation SPEED not static tilt. L2 on gravity-xy makes a standing
        #   tilt cost real reward while the small transient lean a lift genuinely
        #   needs stays cheap (10° -> ~0.03 raw).
        # - feet_slide 0 -> -0.5: feet dragging while IN CONTACT. Prices both halves
        #   of the bench verdict at once: the "кульбит" (front feet re-staging under a
        #   new command = stepping/sliding in place) and the rear feet's horizontal
        #   crawl-instead-of-lift (harness frames 0066/0093: RR/RL slide along the
        #   floor, never leave it).
        # v5 (2026-08-14, v4 ran full 25000 it -- lift never emerged, see
        # TRAINING_STATE.md entry 16:01/20:32 for the full diagnosis): v4's own
        # signal-scaled support_pose exemption fixed the idle-tilt bug it targeted,
        # but STRUCTURALLY it now anchors all 4 legs at idle (was 3 -- FL used to be
        # permanently exempt from the old binary-mask bug) on top of these two NEW
        # universal costs, which fire at idle too -- attempting a lift got
        # measurably more expensive to try than it was in v2/v3, where support_pose
        # already had to be loosened once for exactly this "economic wall" symptom
        # (weight history below). Haircut both by ~30%, not zeroed -- "ровный
        # корпус ВСЕГДА" stays enforced at idle, just less punishing while a leg is
        # actually mid-lift and the other three legitimately need to shift for
        # balance (see leg_lift_support_pose's own -4.0->-3.0 comment: that shift
        # is a PHYSICAL requirement of a real lift, not the cheating this pair was
        # built to catch).
        # v7 (2026-08-16, owner's staging order: "возвращаемся к v2-24999 базе,
        # делаем частями, небольшими порциями, чтобы не запутать PPO"): the
        # v4/v5 universal costs are RETIRED back to v2's zero -- they were built
        # for v4's own goals, and the v4-v6 line demonstrably destroyed the
        # working v2 behavior (correct leg choice, clean lift attempt) without
        # fixing anything. v7 = v2 economy + ONLY the backward-sweep fix package
        # (tolerance 0.08 + proximity gate + base_still), warm-started from
        # v2's own model_24999. Obs layout untouched (45) -- full checkpoint
        # compatibility forward and back.
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.feet_slide.weight = 0

        # -- the lift objective itself
        self.rewards.leg_lift_selected_height = RewTerm(
            func=leg_lift_selected_height,
            weight=6.0,
            params={
                "command_name": "base_velocity",
                # preserve_order=True is load-bearing: without it SceneEntityCfg
                # resolves body_ids in the ARTICULATION's own native order, not
                # FOOT_BODY_NAMES's -- every per-leg function in this file relies on
                # index 0..3 meaning FL,FR,RR,RL specifically (see LEG_DIRECTIONS).
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True),
                # 2026-08-13: hip_cfg + xy_tolerance -- height reward is now GATED by
                # horizontal proximity to the hip (see the function's own docstring
                # for why the earlier side-penalty-only approach didn't stop the
                # backward sweep).
                "hip_cfg": SceneEntityCfg("robot", body_names=HIP_BODY_NAMES, preserve_order=True),
                "target_lift": LIFT_HEIGHT_TARGET,
                "xy_tolerance": LIFT_XY_TOLERANCE,
            },
        )
        self.rewards.leg_lift_support_contact = RewTerm(
            func=leg_lift_support_contact,
            # -3.0 -> -6.0 (v6, 2026-08-15, sequence-frame review of the v5 20700
            # run): under a FORWARD (FL) command the policy lifts FR -- its one
            # practiced leg -- instead. That wrong-leg lift earns ~0 from
            # selected_height (the commanded FL stays planted, clearance 0) but at
            # -3.0 the support-contact price for floating FR was evidently cheap
            # enough to shrug off. Doubled so "reuse the favorite leg" is clearly
            # net-negative and the only profitable clearance is the commanded
            # leg's own. Bounded by construction (<= 3 feet missing).
            # -6.0 -> -3.0 (v7, 2026-08-16): back to the exact v2 value -- the
            # -6.0 belonged to the failed v6 experiment; v7 warm-starts from
            # v2's own 24999, which already picks the CORRECT leg on all four
            # commands (owner-verified live), so the wrong-leg problem this
            # doubling targeted does not exist in the starting policy.
            weight=-3.0,
            params={
                "command_name": "base_velocity",
                # See leg_lift_selected_height's own comment -- preserve_order=True
                # is load-bearing here too, same FL,FR,RR,RL convention.
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
            },
        )
        # New 2026-08-12 (bench on 24999: slow backpedal during the lift -- see
        # leg_lift_base_still's own docstring). Same weight scale as the other
        # tasks' own idle-stillness terms (-2.0..-3.0 on the same v²+w² unit).
        self.rewards.leg_lift_base_still = RewTerm(
            func=leg_lift_base_still,
            weight=-2.0,
            params={"command_name": "base_velocity"},
        )
        # New 2026-08-11 (bench: the lifted leg pulls BACKWARD, not up -- see
        # leg_lift_foot_horizontal's own docstring). Weight matched to the height
        # reward's own +6.0: gaining clearance by sweeping the leg away must never
        # net positive.
        self.rewards.leg_lift_foot_horizontal = RewTerm(
            func=leg_lift_foot_horizontal,
            weight=-6.0,
            params={
                "command_name": "base_velocity",
                "foot_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True),
                "hip_cfg": SceneEntityCfg("robot", body_names=HIP_BODY_NAMES, preserve_order=True),
                "tolerance": LIFT_XY_TOLERANCE,
            },
        )
        # New 2026-08-11 (bench: deep squat during the lift -- see
        # leg_lift_base_height's own docstring). Same weight scale as
        # jump_idle_height's own -8.0 anchor (same L2-on-root-z unit).
        self.rewards.leg_lift_base_height = RewTerm(
            func=leg_lift_base_height,
            weight=-8.0,
            params={"command_name": "base_velocity", "target_height": LIFT_BASE_HEIGHT_TARGET},
        )
        self.rewards.leg_lift_support_pose = RewTerm(
            func=leg_lift_support_pose,
            # -2.0 -> -4.0 (2026-08-11, bench: squat + general support sloppiness --
            # at -2.0 vs the height reward's +6.0 the support stance was too cheap
            # to abandon; leg_lift_base_height now owns the squat specifically, this
            # owns the joint-level stance).
            # -4.0 -> -3.0 (2026-08-13 night, leg_lift v2 run: selected_height
            # plateaued hard at ~4.05-4.10/6.0 for 6 consecutive half-hour checks
            # (~3h) with noise_std still rising, not settling -- not a converged
            # local optimum, a genuine economic wall. Hypothesis: reaching higher
            # clearance under the sharp exp kernel (TRACKING_SIGMA=0.01) forces the
            # three support legs to shift for balance/weight compensation -- a
            # physically necessary part of a HIGH lift, not the cheating this term
            # was built to catch (that's foot_horizontal/base_height's job,
            # untouched here). Loosened, not zeroed -- support legs still owe a
            # recognizable stance, just cheaper to compensate from.
            # -3.0 -> -2.0 (2026-08-14, v4 ran the full 25000 it: selected_height
            # NEVER moved off 3.6-3.9/6.0, flat the entire run -- not a slow climb
            # that needed more time, a genuine standing plateau (see TRAINING_STATE.md
            # 20:32 for the bucketed evidence). Same "economic wall" symptom as the
            # -4.0->-3.0 loosening above, but v4 made the wall taller than v2/v3 had
            # it WITHOUT changing this number: it now anchors all 4 legs at idle
            # (was 3) plus the two new universal costs below -- continuing the same
            # lever that already worked twice.
            # -2.0 -> -4.0 (v7, 2026-08-16): back to the exact v2 value. Both
            # loosenings above belonged to the v3-v6 line that never reproduced
            # v2's clean behavior; v7 restores the economy the working 24999 was
            # actually trained under (the ONE deliberate carry-over: the
            # signal-scaled exemption replacing v2's binary mask -- pure idle-time
            # bugfix, identical math during an active lift, prevents the
            # bench-proven FL-idle-drift/кульбит channel).
            weight=-4.0,
            params={
                "command_name": "base_velocity",
                # preserve_order=True -- this function's .view(N, 4, 3) reshape
                # assumes joint_ids comes back grouped FL/FR/RR/RL in exactly
                # LEG_JOINT_NAMES_ORDERED's order, not the articulation's native one.
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES_ORDERED, preserve_order=True),
            },
        )

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeB2LegLiftRoughEnvCfg":
            self.disable_zero_weight_rewards()
