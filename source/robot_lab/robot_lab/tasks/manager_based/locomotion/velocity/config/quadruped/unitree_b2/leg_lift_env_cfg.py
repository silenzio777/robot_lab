# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Single-leg lift ("трипод") for B2 -- a deliberately simple test task (user, 2026-08-09:
"он простой, тестовый, я хочу понять что можно делать через RL а что придётся через IL").

v9 (2026-08-20, task relayed by claude-tg-base after v8.0-v8.5 all failed to
converge, `train_research/LEG_LIFT_V9_TASK.md`): SYSTEMIC redesign, not another
weight tweak.

DIAGNOSIS: every v8.x symptom (squatting, -16 deg pitch, idle with two feet
airborne, curriculum pinned at its floor for a FULL 20000-iteration budget) was
downstream of ONE decision, made back in v7 and never revisited: this file
zeroed rough_env_cfg's own `stand_still_without_cmd`/`joint_pos_penalty`
(the STOCK "hold default pose" recipe walk/crawl stand rock-solid on) and
reinvented standing from scratch with a custom `leg_lift_support_pose`
anchor. Every v8.x fix (support_pose weight, always-on base_height,
flat_orientation_l2 strengthening, an absolute foot-height floor) was a patch
on top of that reinvention, not a fix to the reinvention itself. For a task
where 3 of 4 legs and ~all of idle time should behave EXACTLY like the stock
"stand still in default pose" recipe already does, throwing that recipe away
and rebuilding a weaker version of it from scratch was the actual bug.

v9 design: stock standing package back, masked (the SELECTED leg's 3 joints
exempted from the stock anchors while a lift is actively commanded -- one
line of masking inside otherwise-stock functions, not a reinvention) +
exactly THREE custom terms for the lift objective itself (absolute foot
height, feet-on-air, air-time -- handstand-proportioned 10/5/5), and NOTHING
else custom. Removed entirely (all were patches on the reinvented
foundation, not needed once the real foundation is back): leg_lift_support_
pose, leg_lift_foot_height_floor, leg_lift_base_height, leg_lift_base_still,
leg_lift_com_over_support, leg_lift_joint_fold, leg_lift_foot_horizontal,
leg_lift_support_contact, and the old relative-clearance+proximity-gated
leg_lift_selected_height. flat_orientation_l2/feet_slide reverted to
rough_env_cfg's own stock values (0 for both -- confirmed walk_env_cfg.py
itself never overrides flat_orientation_l2 either; the "ровный корпус
ВСЕГДА" pressure v8.4 tried to buy with -10.0 is what the restored stock
joint-pose anchors are supposed to deliver for free, same as they do for
walk).

Height reference: ABSOLUTE world-Z of the calf body (FOOT_BODY_NAMES,
unchanged convention from v2-v8 -- NOT the URDF's separate `*_foot` sphere
link, which the local MuJoCo bench doesn't expose as its own body at all
{net_forces_w/mj_name2id both return nothing for "FL_foot" on this bench's
compiled model, the fixed joint gets merged into the calf at compile time},
so switching now would desync every piece of bench verification tooling
built this whole session for a difference that's small once we're talking
about genuine multi-cm lifts). Curriculum/sigma recalibrated by direct
MuJoCo FK in these terms -- see LIFT_HEIGHT_INIT/MAX/FOOT_REST_Z's own
comments for the numbers and the pre-flight check that verified them.

Also fixes a real (if previously inert) bug in `_direction_to_leg_mask`:
below CMD_ACTIVE_THRESHOLD the one-hot used to still mark FL selected (an
argmax-style default), so anything reading the mask without ALSO gating on
signal would silently credit FL at idle. v8's own consumers all happened to
multiply by signal/risen (itself ~0 at idle) so this stayed dormant, but
v9's air-time term gates on the mask's own `active` flag directly -- fixed
by zeroing the whole mask row when inactive instead of leaving a phantom
FL selection.

Command interface (v8, kept unchanged -- not implicated in any v8.x
failure): 2-slot (lin_vel_x, lin_vel_y) cmd_vel, magnitude doubles as the
old "signal". Direction mapping (owner's spec): forward(+x)->FR,
right(-y)->RR, back(-x)->RL, left(+y)->FL. See b2_leg_lift_driver.py (bench
driver, kept in sync by hand) for the raw-stick side of this.
"""

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeB2RoughEnvCfg

# Command timing -- unchanged since v7, never implicated in any failure.
IDLE_TIME_RANGE = (1.5, 3.0)
RISE_DURATION = 0.6
HOLD_TIME_RANGE = (1.5, 3.0)
DESCEND_DURATION = 0.6

# -- v9 Rudin-style per-env height curriculum, recalibrated in ABSOLUTE
# calf-body world-Z terms (see module docstring for why calf-body, not the
# URDF's separate foot/toe link) --
#
# FOOT_REST_Z: MuJoCo FK at the default joint pose (thigh=0.8, calf=-1.5,
# hip=0, matching unitree.py's own default_joint_pos) -- measured directly
# on this bench, 0.336m. This is what "foot at rest, no lift attempted"
# actually reads as in the units the reward tracks; NOT the "~0.04m" a true
# ground-contact toe would read (claude-tg-base's own original estimate,
# written in foot-not-calf terms -- flagged and corrected before this file
# was written, see TRAIN_RESEARCH.md).
FOOT_REST_Z = 0.336  # m
# LIFT_HEIGHT_INIT: thigh~1.0-1.1 rad by the same FK sweep (hip=0,
# calf-independent -- calf_joint doesn't move the calf body's own position,
# verified in v8.2's own investigation and unchanged physics here) --
# 0.45m, a genuine ~11cm lift above rest, achievable but not trivial.
LIFT_HEIGHT_INIT = 0.45  # m, level-0 curriculum target
LIFT_HEIGHT_MIN = 0.40  # m, floor a failing env can regress to -- still a
# real, if modest, lift (never regress all the way back to "don't bother")
# LIFT_HEIGHT_MAX: thigh~1.5 rad -> calf_z~0.555m by the same FK sweep --
# close to LIFT_BASE_HEIGHT_TARGET-equivalent root height (0.53m, rough's
# own standing target), i.e. "foot roughly at torso level", matching the
# owner's own "стопа до уровня корпуса" spec without chasing the thigh
# joint's own kinematic singularity (see the retired THIGH_FOLD_TARGET
# history below this block for why overshooting that FK curve is a real
# risk, not a hypothetical one).
LIFT_HEIGHT_MAX = 0.55  # m
LIFT_HEIGHT_LEVEL_STEP = 0.02  # m, per-cycle bump on success -- unchanged
# magnitude from v8 (never the part of the design that was wrong)
LIFT_HEIGHT_LEVEL_DOWN = 0.006  # m, gentler per-cycle regression on failure
# v9 FIX: v8's success criterion was "within 3cm for the ENTIRE hold
# phase" (1.5-3.0s, randomized) -- a strict, all-or-nothing bar that never
# once fired in 20000 iterations across v8.3-v8.5 (curriculum pinned at its
# floor the whole run, see TRAINING_STATE.md's own v8.5 postmortem).
# claude-tg-base's task doc: soften to "5cm tolerance, held for >=1.5s
# cumulative during the hold" -- a duration bar, not a whole-hold bar, so a
# few noisy steps don't zero out an otherwise-good hold.
LIFT_HEIGHT_SUCCESS_TOL = 0.05  # m
LIFT_HEIGHT_SUCCESS_DURATION = 1.5  # s, cumulative in-tolerance time needed

TRACKING_SIGMA = 0.01  # m^2 -- reused for the new absolute-height kernel
# too; see the pre-flight check in TRAIN_RESEARCH.md/TRAINING_STATE.md for
# the numeric verification this gives a live (non-e-16) gradient at rest
# against LIFT_HEIGHT_INIT (reward~0.27 at foot_z=FOOT_REST_Z, target=0.45).

# -- v8.0-v8.2's retired joint-fold anchor, kept as a comment for the
# historical record (not deleted -- same "keep the failed attempt's
# reasoning" discipline this file's own history already uses). v9 removes
# leg_lift_joint_fold ENTIRELY, not just its thigh half: the whole
# secondary-anchor idea is superseded by the stock joint_pos_penalty/
# stand_still_without_cmd package now covering pose for the 3 support legs,
# and the selected leg's own pose falls out of leg_lift_foot_height's own
# absolute-height objective instead of a separate joint-space target.
# THIGH_FOLD_TARGET = 3.0 rad, CALF_FOLD_TARGET = -2.5 rad -- see v8.2's
# own git history for the full FK-conflict diagnosis (thigh angles that
# bring the foot horizontally near the hip ALSO overshoot vertical
# clearance to ~0.55-0.57m at the SAME time xy is small, so a single
# fixed joint target could never satisfy both an xy-proximity gate and a
# modest height target at once -- part of why v9 drops the proximity gate
# too, see leg_lift_foot_height's own docstring).

# Direction mapping (owner's spec, unchanged since v8) -- forward=FR,
# right=RR, back=RL, left=FL. Order below is (dx, dy) per direction, used
# ONLY to pick which canonical unit vector a newly-resampled cycle
# commands; leg SELECTION itself is computed from the live command
# vector's own sign/dominance in _selected_leg_mask, not by table lookup.
CMD_DIRECTIONS = ((1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0))  # fwd,right,back,left
CMD_ACTIVE_THRESHOLD = 0.1  # |cmd| below this reads as "no leg selected"

# Explicit, ORDERED (FL,FR,RR,RL) name lists -- every per-leg-shaped reward
# function below relies on this exact order matching _selected_leg_mask's
# own index convention.
FOOT_BODY_NAMES = ["FL_calf", "FR_calf", "RR_calf", "RL_calf"]
LEG_JOINT_NAMES_ORDERED = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]  # fmt: skip


class LegLiftCommand(CommandTerm):
    """Cycle: idle (four legs) -> rise -> hold (one leg lifted) -> descend -> resample,
    same clock trick as jump's JumpPulseCommand / rear_stand's RearStandCommand.

    2-slot (lin_vel_x, lin_vel_y) command, magnitude doubles as `.signal` (0
    idle, ramps to 1 held, ramps back down) -- unchanged from v8.

    v9.4 FIX (2026-08-21, base's design after the gravity-feedforward action
    offset failed validation -- see leg_lift_env_cfg.py's own history and
    train_research/TRAINING_STATE.md's 17:3x-18:2x entries): widened to a
    3rd, always-zero slot ([lin_vel_x, lin_vel_y, 0.0], matching stock
    walk's [vx, vy, wz] width) purely so this task's actor/critic
    observation widths match unitree_b2_rough's own (45/235, verified
    against unitree_b2_rough/2026-08-01_13-48-23/model_5000.pt's own
    state_dict shapes before this run) -- enabling a cross-task warm-start
    resume from that checkpoint. The root problem was never the reward
    weights: walk already holds default_joint_pos rock-solid at cmd=0 under
    the exact same kp=160 gravity load leg_lift does, it just never gets a
    chance to LEARN that precompensation from scratch because leg_lift's
    idle state has "do nothing" as an unusually strong local optimum (walk
    is always moving, so it never falls into that optimum in the first
    place). Resuming from a network that already solved this exact
    sub-problem sidesteps it entirely -- no offset, no reward escalation.

    Owns the Rudin-style per-env height curriculum (`height_target`,
    reworked for v9 into a DURATION-based success criterion --
    `_hold_intol_time`/`_hold_entered` -- instead of v8's whole-hold boolean
    AND, see LIFT_HEIGHT_SUCCESS_DURATION's own comment for why)."""

    cfg: "LegLiftCommandCfg"

    def __init__(self, cfg: "LegLiftCommandCfg", env) -> None:
        super().__init__(cfg, env)
        n = self.num_envs
        # 3rd column (wz-shaped slot) stays exactly 0.0 forever -- _update_command
        # below only ever writes columns 0:2, matching walk's [vx, vy, wz=0] shape
        # for the warm-start resume (see class docstring's v9.4 note).
        self._command = torch.zeros(n, 3, device=self.device)
        self.signal = torch.zeros(n, device=self.device)
        self.direction = torch.zeros(n, 2, device=self.device)
        self.idle_duration = torch.zeros(n, device=self.device)
        self.hold_duration = torch.zeros(n, device=self.device)
        self.cycle_duration = torch.zeros(n, device=self.device)
        self._directions = torch.tensor(cfg.directions, dtype=torch.float, device=self.device)

        # -- height curriculum state --
        self.height_target = torch.full((n,), LIFT_HEIGHT_INIT, device=self.device)
        self._hold_intol_time = torch.zeros(n, device=self.device)  # cumulative
        # in-tolerance seconds during the CURRENT hold phase
        self._hold_entered = torch.zeros(n, dtype=torch.bool, device=self.device)

        # Cached body indices for the curriculum's own height check (FL,FR,RR,RL
        # order, matching FOOT_BODY_NAMES) -- resolved once here instead of going
        # through SceneEntityCfg machinery, since this is purely internal
        # bookkeeping, not a manager-registered term.
        body_names = self._env.scene["robot"].body_names
        self._foot_ids = [body_names.index(n) for n in FOOT_BODY_NAMES]

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        self.metrics["lift_signal"] = self.signal.clone()
        self.metrics["height_target"] = self.height_target.clone()

    def _resample_command(self, env_ids):
        idle = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.idle_time_range)
        hold = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.hold_time_range)
        self.idle_duration[env_ids] = idle
        self.hold_duration[env_ids] = hold
        self.cycle_duration[env_ids] = idle + self.cfg.rise_duration + hold + self.cfg.descend_duration
        self.time_left[env_ids] = self.cycle_duration[env_ids]
        dir_idx = torch.randint(0, self._directions.shape[0], (len(env_ids),), device=self.device)
        self.direction[env_ids] = self._directions[dir_idx]

        # -- height curriculum: judge the cycle that just ended BEFORE resetting
        # the trackers for the new one. `_hold_entered` guards envs on their very
        # first-ever resample (no hold phase has happened yet, nothing to judge).
        judge = self._hold_entered[env_ids]
        succeeded = judge & (self._hold_intol_time[env_ids] >= LIFT_HEIGHT_SUCCESS_DURATION)
        failed = judge & ~(self._hold_intol_time[env_ids] >= LIFT_HEIGHT_SUCCESS_DURATION)
        up_ids = env_ids[succeeded]
        down_ids = env_ids[failed]
        if len(up_ids) > 0:
            self.height_target[up_ids] = (self.height_target[up_ids] + LIFT_HEIGHT_LEVEL_STEP).clamp(max=LIFT_HEIGHT_MAX)
        if len(down_ids) > 0:
            self.height_target[down_ids] = (self.height_target[down_ids] - LIFT_HEIGHT_LEVEL_DOWN).clamp(min=LIFT_HEIGHT_MIN)
        self._hold_intol_time[env_ids] = 0.0
        self._hold_entered[env_ids] = False

    def _update_command(self):
        elapsed = self.cycle_duration - self.time_left
        rise_start = self.idle_duration
        hold_start = rise_start + self.cfg.rise_duration
        descend_start = hold_start + self.hold_duration
        rising = ((elapsed - rise_start) / self.cfg.rise_duration).clamp(0.0, 1.0)
        descending = ((elapsed - descend_start) / self.cfg.descend_duration).clamp(0.0, 1.0)
        self.signal = rising - descending  # 0 idle, ramps up, holds 1, ramps down
        active = (elapsed >= rise_start).float().unsqueeze(-1)
        self._command[:, 0:2] = self.direction * active * self.signal.unsqueeze(-1)

        # -- height curriculum: accumulate in-tolerance TIME while actually
        # holding (v9: duration-based, not a whole-hold boolean AND -- see
        # LIFT_HEIGHT_SUCCESS_DURATION's own comment). Uses the SAME absolute
        # calf-body height leg_lift_foot_height itself tracks.
        in_hold = (elapsed >= hold_start) & (elapsed < descend_start)
        if torch.any(in_hold):
            asset = self._env.scene["robot"]
            foot_z = asset.data.body_pos_w[:, self._foot_ids, 2]
            mask = _direction_to_leg_mask(self._command)
            selected_z = (foot_z * mask).sum(dim=1)
            within_tol = torch.abs(selected_z - self.height_target) < LIFT_HEIGHT_SUCCESS_TOL
            self._hold_intol_time = self._hold_intol_time + within_tol.float() * in_hold.float() * self._env.step_dt
            self._hold_entered = self._hold_entered | in_hold


@configclass
class LegLiftCommandCfg(CommandTermCfg):
    class_type: type = LegLiftCommand
    resampling_time_range: tuple[float, float] = (4.0, 8.0)  # nominal; overwritten per cycle
    idle_time_range: tuple[float, float] = IDLE_TIME_RANGE
    rise_duration: float = RISE_DURATION
    hold_time_range: tuple[float, float] = HOLD_TIME_RANGE
    descend_duration: float = DESCEND_DURATION
    directions: tuple = CMD_DIRECTIONS
    debug_vis: bool = False


def _direction_to_leg_mask(command: torch.Tensor) -> torch.Tensor:
    """[N,4] one-hot (FL,FR,RR,RL order) from a live (lin_vel_x, lin_vel_y) command,
    by dominant-axis + sign (owner's spec: "выбор ноги по квадранту/знаку
    доминирующей оси, порог |cmd|>0.1").

    v9 FIX: below CMD_ACTIVE_THRESHOLD this now returns an ALL-ZERO row (no
    leg selected) instead of v8's own argmax-style default-to-FL(index 0).
    That default was dormant in v8 (every consumer also multiplied by
    signal/risen, itself ~0 exactly when the default could fire) but v9's
    leg_lift_foot_air_time gates on the mask's own `active` state directly --
    a phantom FL selection would have silently credited FL at idle.

    Mapping (owner's spec): +x(fwd)->FR(1), -y(right)->RR(2), -x(back)->RL(3),
    +y(left)->FL(0)."""
    vx, vy = command[:, 0], command[:, 1]
    x_dominant = torch.abs(vx) >= torch.abs(vy)
    active = torch.linalg.norm(command, dim=-1) > CMD_ACTIVE_THRESHOLD
    idx = torch.zeros(command.shape[0], dtype=torch.long, device=command.device)
    idx = torch.where(x_dominant & (vx > 0), torch.full_like(idx, 1), idx)  # forward -> FR
    idx = torch.where(x_dominant & (vx <= 0), torch.full_like(idx, 3), idx)  # back -> RL
    idx = torch.where(~x_dominant & (vy <= 0), torch.full_like(idx, 2), idx)  # right -> RR
    idx = torch.where(~x_dominant & (vy > 0), torch.full_like(idx, 0), idx)  # left -> FL
    mask = torch.zeros(command.shape[0], 4, device=command.device, dtype=command.dtype)
    mask.scatter_(1, idx.unsqueeze(-1), 1.0)
    mask = mask * active.float().unsqueeze(-1)  # v9 FIX: zero the whole row when inactive
    return mask


def _selected_leg_mask(env, command_name: str) -> torch.Tensor:
    """Reward-function-facing wrapper around _direction_to_leg_mask -- fetches the
    live command tensor from the named command term."""
    term = env.command_manager.get_term(command_name)
    return _direction_to_leg_mask(term.command)


def leg_lift_masked_stand_still_without_cmd(
    env, command_name: str, command_threshold: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """v9. Same as the STOCK `mdp.stand_still_without_cmd` (L1 joint-pos-vs-
    default penalty, gated on low command AND on being upright) -- the ONE
    change is masking the SELECTED leg's 3 joints out while a lift is
    actively commanded, so this term can't fight the very lift it's meant to
    otherwise prevent. In practice this term is ALREADY zero whenever the
    command exceeds `command_threshold` (the stock gate), so the mask mostly
    matters at the RISE/DESCEND ramp edges where signal is nonzero but the
    raw command magnitude can still dip under threshold -- kept for
    correctness/symmetry with leg_lift_masked_joint_pos_penalty below, where
    the mask is load-bearing."""
    asset = env.scene[asset_cfg.name]
    diff_angle = torch.abs(
        asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    )
    diff_angle = diff_angle.view(diff_angle.shape[0], 4, 3)
    term = env.command_manager.get_term(command_name)
    mask = _selected_leg_mask(env, command_name)
    exemption = mask * term.signal.unsqueeze(-1)
    diff_angle = diff_angle * (1.0 - exemption).unsqueeze(-1)
    reward = torch.sum(diff_angle, dim=(1, 2))
    reward = reward * (torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) < command_threshold)
    reward = reward * (torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7)
    return reward


def leg_lift_masked_joint_pos_penalty(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """v9. Same as the STOCK `mdp.joint_pos_penalty` -- masked the same way as
    leg_lift_masked_stand_still_without_cmd above. THIS mask is load-bearing
    (unlike the still-without-cmd sibling): the stock function's own "moving"
    branch (cmd>threshold OR body moving) uses the FULL joint-deviation norm
    with no gate at all, so without masking, lifting the selected leg would
    be directly penalized by this term at exactly the moment it's supposed
    to move -- the stock recipe was built assuming EVERY joint deviation
    during "moving" is gait-related and fine, which isn't true here."""
    asset = env.scene[asset_cfg.name]
    cmd = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    diff = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    diff = diff.view(diff.shape[0], 4, 3)
    term = env.command_manager.get_term(command_name)
    mask = _selected_leg_mask(env, command_name)
    exemption = mask * term.signal.unsqueeze(-1)
    diff = diff * (1.0 - exemption).unsqueeze(-1)
    running_reward = torch.linalg.norm(diff.reshape(diff.shape[0], 12), dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward = reward * (torch.clamp(-asset.data.projected_gravity_b[:, 2], 0, 0.7) / 0.7)
    return reward


def leg_lift_foot_height(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """v9 NEW (1 of 3 custom terms, dominant, weight 10). exp-tracking of the
    SELECTED leg's ABSOLUTE world-Z (calf-body convention, see module
    docstring) toward the per-env Rudin-curriculum target, ramped from
    FOOT_REST_Z (idle) to `term.height_target` (fully commanded) by signal --
    same ramp idiom every other B2 skill's own idle->target anchor already
    uses.

    v8's own `leg_lift_selected_height` tracked clearance RELATIVE to the
    other three feet's average height instead -- gameable (v8.2's diagnosed
    exploit: crouch the three support legs, which raises "clearance" exactly
    as much as lifting the selected foot would) and additionally gated by
    horizontal proximity to the hip (v7's own anti-sweep fix). v9 drops BOTH:
    absolute height can't be gamed by moving OTHER legs (the restored stock
    joint-pose anchors are what keeps the support legs honest now, not this
    term), and the proximity gate is gone because claude-tg-base's review
    traced the entire v8.0-v8.1 stall to that gate multiplying the height
    kernel to numerical zero at every thigh angle that could also satisfy
    it (see the retired THIGH_FOLD_TARGET history above) -- a sharp
    multiplicative gate is exactly the kind of thing that silently kills a
    gradient, and this file has now paid for that lesson twice."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # [N,4], FL,FR,RR,RL order
    mask = _selected_leg_mask(env, command_name)
    selected_z = (foot_z * mask).sum(dim=1)
    target = FOOT_REST_Z + (term.height_target - FOOT_REST_Z) * term.signal
    return torch.exp(-torch.square(selected_z - target) / TRACKING_SIGMA)


def leg_lift_foot_on_air(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """v9 NEW (2 of 3, weight 5). Positive one-shot bonus the instant the
    SELECTED leg leaves the ground -- ported from unitree_a1_handstand's own
    handstand_feet_on_air (v8's leg_lift_feet_on_air, unchanged logic, just
    renamed to match the task doc's own singular "foot" naming and the v9
    mask fix below it depends on). Gated on signal>0.5 (sustained lift, not
    rise-ramp noise)."""
    term = env.command_manager.get_term(command_name)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids].float()  # [N,4]
    mask = _selected_leg_mask(env, command_name)
    reward = (first_air * mask).sum(dim=1)
    risen = (term.signal > 0.5).float()
    return reward * risen


def leg_lift_foot_air_time(env, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """v9 NEW (3 of 3, weight 5). Rewards sustained air time on the SELECTED
    leg past `threshold` -- ported from unitree_a1_handstand's own
    handstand_feet_air_time (v8's leg_lift_feet_air_time, unchanged logic,
    renamed to match). Gate is the mask's own `active` state (v9's
    _direction_to_leg_mask fix, see its own docstring) rather than a
    separate `risen` check -- closes the v8 FL-bias this exact term would
    otherwise have been the first to actually expose (v8's `feet_on_air`
    also gated on signal>0.5, itself ~0 at idle, so the old default-to-FL
    mask never actually fired there either -- but relying on that
    coincidence twice was the bug, not the fix)."""
    term = env.command_manager.get_term(command_name)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward_all = (last_air_time - threshold) * first_contact.float()
    mask = _selected_leg_mask(env, command_name)
    return (reward_all * mask).sum(dim=1)


class UnitreeB2LegLiftRoughEnvCfg(UnitreeB2RoughEnvCfg):
    """See module docstring -- a deliberately simple test skill, not a production one."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Flat ground -- unchanged reasoning since v7: the trick is plenty hard
        # without rough terrain on top.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        # The lift command replaces the velocity command wholesale (2-slot
        # cmd_vel, unchanged from v8).
        self.commands.base_velocity = LegLiftCommandCfg()
        self.curriculum.command_levels = None

        # -- ONLY zeroings this file makes (v9): no velocity command to
        # track, and the task is inherently asymmetric. Everything else
        # stays at rough_env_cfg's own stock B2 values -- see module
        # docstring for why v8's additional zeroings/overrides
        # (stand_still_without_cmd, joint_pos_penalty, flat_orientation_l2,
        # feet_slide) are GONE, not just re-tuned.
        self.rewards.track_lin_vel_xy_exp.weight = 0
        self.rewards.track_ang_vel_z_exp.weight = 0
        self.rewards.joint_mirror.weight = 0

        # Not a periodic gait -- feet_height/feet_height_body retired in favor
        # of leg_lift_foot_height, which is specific to the one commanded leg.
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height_body.weight = 0

        # -- v9: the STOCK standing package, masked (see the two functions'
        # own docstrings) -- restored at rough_env_cfg's own B2 weights
        # (-2.0/-1.0), not reinvented. This the whole point of the redesign:
        # walk/crawl hold default pose rock-solid on exactly these two terms
        # at exactly these weights; leg_lift gets the same foundation, only
        # exempting the one leg that's SUPPOSED to move.
        self.rewards.stand_still_without_cmd = RewTerm(
            func=leg_lift_masked_stand_still_without_cmd,
            weight=-2.0,
            params={
                "command_name": "base_velocity",
                "command_threshold": 0.1,
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES_ORDERED, preserve_order=True),
            },
        )
        self.rewards.joint_pos_penalty = RewTerm(
            func=leg_lift_masked_joint_pos_penalty,
            weight=-1.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES_ORDERED, preserve_order=True),
                "stand_still_scale": 5.0,
                "velocity_threshold": 0.5,
                "command_threshold": 0.1,
            },
        )

        # flat_orientation_l2, feet_slide, upward: DELIBERATELY left untouched
        # at whatever rough_env_cfg.py already set (0, 0, 3.0 respectively) --
        # "стоковые веса", not a v9 override. See module docstring for why
        # v8.4's -10.0 orientation strengthening is gone, not increased
        # further.

        # -- the lift objective: exactly three custom terms, handstand-
        # proportioned weights (10/5/5) -- see each function's own docstring.
        #
        # STAGE 0 (2026-08-21, base's design after v9.4's warm-start it+100
        # gate failed and BOTH follow-up hypotheses -- mass mismatch,
        # actuator sim2sim -- were checked and refuted, see TRAINING_STATE.md
        # 18:25-19:10 entries): all three weights forced to 0 here (which
        # `disable_zero_weight_rewards()` below then drops from the reward
        # manager entirely -- these terms don't run at all this stage, no
        # observation-width impact, only RewardsCfg is touched).
        #
        # Root reframing: nobody ever confirmed walk's RL policy could
        # genuinely STAND at cmd=0 in the first place. `rel_standing_envs=
        # 0.02` in velocity_env_cfg.py's own UniformVelocityCommandCfg
        # (verified directly, not from memory) means walk trains on a
        # zero-command episode only ~2% of the time -- and on the real
        # robot, standing between maneuvers is a separate classical-PD
        # FixStand state, never this RL policy's job at all. The bench's
        # -16deg idle-pitch measurement on a COMPLETELY untouched walk
        # checkpoint (Test W, this session) is therefore likely the
        # policy's genuine, never-fixed cmd=0 equilibrium, not a bench/mass/
        # actuator artifact (both of those were independently ruled out).
        # leg_lift is the first task in this lab where genuine idle standing
        # is actually load-bearing -- Stage 0 asks it to learn JUST that,
        # nothing else, warm-started from walk's general locomotion skill
        # but with the novel lift objective entirely absent so there's no
        # large new gradient competing with re-learning to stand still.
        # Gate: bench-idle pitch <3 deg (base's threshold). Stage 1 (already
        # designed, not yet run) ramps these three weights back 0->10/5/5
        # from whatever checkpoint clears this gate.
        self.rewards.leg_lift_foot_height = RewTerm(
            func=leg_lift_foot_height,
            weight=0.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True),
            },
        )
        self.rewards.leg_lift_foot_on_air = RewTerm(
            func=leg_lift_foot_on_air,
            weight=0.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
            },
        )
        self.rewards.leg_lift_foot_air_time = RewTerm(
            func=leg_lift_foot_air_time,
            weight=0.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
                # Hold windows are 1.5-3.0s (HOLD_TIME_RANGE) -- 1.0s threshold
                # asks for a genuinely sustained lift, not a touch-and-go.
                # Unchanged value from v8 (never implicated in any failure).
                "threshold": 1.0,
            },
        )

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeB2LegLiftRoughEnvCfg":
            self.disable_zero_weight_rewards()
