# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Single-leg lift ("трипод") for B2 -- a deliberately simple test task (user, 2026-08-09:
"он простой, тестовый, я хочу понять что можно делать через RL а что придётся через IL").

v8 (2026-08-19, task relayed by claude-tg-base, `train_research/LEG_LIFT_V8_TASK.md`):
REDESIGN from scratch on top of v2/v7's proven skeleton (support/height/pose economy
kept, see the individual functions below -- most are unchanged from v7), adding the
four things v7 structurally lacked:
  1. a POSITIVE air package on the selected leg (feet_on_air/feet_air_time, ported
     from unitree_a1_handstand's own rewards.py, masked per-env instead of a static
     body subset since here the "which leg" varies per env/command);
  2. a CoM-over-support-TRIANGLE term (centroid of the three support feet, not a
     fixed pair -- see leg_lift_com_over_support);
  3. a secondary joint-angle anchor on the lifted leg's thigh+calf toward a "foot
     folded up near the body" pose (leg_lift_joint_fold), decoupled from pure
     clearance the same way rear_stand_rear_leg_extension decoupled knee angle from
     root height;
  4. a Rudin-style PER-ENV game curriculum on the lift-height target (height_target
     lives on the command term itself, bumped per env based on that env's own
     previous-cycle success/failure -- not a global schedule).

COMMAND INTERFACE CHANGED (owner's explicit spec this time, not a v7-era holdover):
2 slots now, a genuine (lin_vel_x, lin_vel_y)-shaped cmd_vel instead of v7's
[lift_signal, dir_x, dir_y] -- the ramped magnitude of the vector itself doubles as
the old "signal" (kept as the `.signal` attribute below for every reward function
that still reads it), so no separate slot is needed. Leg selection reads the same
observable vector: dominant axis + its sign, threshold |cmd|>0.1 -- exactly the rule
a real joystick-driven bench has to apply too, see b2_leg_lift_driver.py (kept in
sync by hand, same convention as every other B2 skill driver in this repo).

Direction mapping is the OWNER'S NEW spec for v8 (rotated one quadrant from v2/v7's
own forward=FL mapping -- do not reuse the old LEG_DIRECTIONS table for this):
  forward (+x) -> FR   right (-y) -> RR   back (-x) -> RL   left (+y) -> FL
Sign convention: standard body-frame (+x forward, +y left) -- verified against this
bench's own raw stick axes via b2_leg_lift_driver.py's predecessor (jump family
already established vy>0=LEFT on this hardware), so no sign flip is needed anywhere
in the chain: raw stick -> [vx, vy] -> training command, all the same signs.

Breaks v7's "45-obs, forward/backward warm-start compatible" property (command
width 3->2 means total obs 45->44) -- deliberate, not an oversight: the command
INTERFACE itself changed, so there is nothing meaningful left to warm-start from.
Trained from scratch, same as jump v6 and rear_stand v7 this same session.
"""

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeB2RoughEnvCfg

# Command timing -- unchanged from v7 (never diagnosed as a problem; only the
# reward economy and command interface are being redesigned this round).
IDLE_TIME_RANGE = (1.5, 3.0)
RISE_DURATION = 0.6
HOLD_TIME_RANGE = (1.5, 3.0)
DESCEND_DURATION = 0.6

# -- Rudin-style per-env height curriculum (v8, new) --
# Starts at v7's old fixed LIFT_HEIGHT_TARGET (0.15) as the level-0 floor -- a value
# already proven reachable by the v2/v7 lineage, so level 0 should not itself be a
# struggle. MAX derived from thigh-link geometry (b2 URDF: FL_thigh_joint -> FL_calf
# origin offset is 0.35m along the thigh) -- folding the whole leg up toward the hip
# can geometrically put the foot within roughly one thigh-length of the hip's own
# height, so 0.35m is the kinematic ceiling for "foot near body" as a CLEARANCE
# figure; 0.30 leaves the same kind of headroom-from-singularity margin
# STAND_HEIGHT_TARGET/REAR_LEG_EXTENSION_TARGET both used (an env that reaches 0.30m
# consistently has essentially solved the task; there is little value in chasing the
# literal kinematic limit). FIRST GUESS on the level step/tolerance/floor -- expect
# postmortem-driven recalibration, same epistemic status as every other first-guess
# constant in this file's own history.
LIFT_HEIGHT_INIT = 0.15  # m, level-0 target (== v7's old fixed value)
LIFT_HEIGHT_MIN = 0.05  # m, floor a failing env can regress to, never below
LIFT_HEIGHT_MAX = 0.30  # m, ceiling -- see thigh-geometry comment above
LIFT_HEIGHT_LEVEL_STEP = 0.02  # m, per-cycle bump on success
LIFT_HEIGHT_LEVEL_DOWN = 0.006  # m, gentler per-cycle regression on failure (~30% of
# the up-step -- Rudin-style asymmetry so one bad cycle doesn't erase several good
# ones; still genuinely two-way, not monotonic-only)
LIFT_HEIGHT_SUCCESS_TOL = 0.03  # m, |clearance - target| must stay under this for
# EVERY step of the hold phase to count as a success this cycle (a strict, binary,
# whole-hold criterion -- deliberately simple over a fractional/time-weighted one,
# see leg_lift_env_cfg's own module docstring point 4)

TRACKING_SIGMA = 0.01  # m^2, sharp -- clearance error is naturally small-scale (meters)
LIFT_XY_GATE_SIGMA = 0.004  # m^2, ~2.5x sharper than TRACKING_SIGMA (see
# leg_lift_selected_height's own docstring for the full "why a separate constant"
# reasoning, unchanged from v7)
LIFT_XY_TOLERANCE = 0.08  # m, unchanged from v7 -- proven fix for the backward-sweep
# cheat (see leg_lift_selected_height/leg_lift_foot_horizontal docstrings)
LIFT_BASE_HEIGHT_TARGET = 0.53  # m, unchanged from v7 -- rough's own standing target

# -- v8 joint-fold anchor targets (new) --
# "Оба сустава до уровня корпуса" (owner's spec) -- fold thigh+calf so the foot
# tucks up near the hip, decoupled from pure world-Z clearance the same way
# rear_stand_rear_leg_extension decoupled knee angle from root height (see that
# function's own docstring for the precedent).
#
# Derived via the SAME forward-kinematics discipline STAND_HEIGHT_TARGET/
# REAR_LEG_EXTENSION_TARGET both used (not copied from an untested guess): b2 URDF,
# FL_thigh_joint rotates about local +Y, and its child (the calf's own origin) sits
# at local (0, 0, -0.35) in the thigh's own frame at thigh_joint=0. Under a +Y
# rotation by q1, that point moves to (-0.35*sin(q1), *, -0.35*cos(q1)) in the
# thigh-origin frame -- at the default standing value (q1=0.8) this is
# (-0.25, *, -0.24), i.e. hanging down-and-back, consistent with a normal standing
# leg. Increasing q1 well past 0.8 rotates the same point up-and-back; at q1~pi/2
# the thigh is roughly horizontal, and further increase starts lifting the knee
# ABOVE the hip. thigh_joint's own URDF range is [-0.94, 4.69] -- unusually wide,
# clearly built to allow exactly this fold-up range (nowhere near it for a normal
# gait). Picked THIGH_FOLD_TARGET=2.4 rad (~137 deg) as a first-guess mid-fold: past
# horizontal, meaningfully "up", well short of the 4.69 singularity (same
# leave-headroom discipline as every other target constant here).
#
# calf_joint range is [-2.82, -0.43] (default -1.5); more-negative = knee bent
# tighter (shank folded back toward the thigh). CALF_FOLD_TARGET=-2.5 folds the
# shank most of the way toward its own limit, leaving ~0.32 rad margin from -2.82 --
# needed so the whole assembly (thigh rotated up + calf folded back) stays compact
# enough for the foot to actually end up NEAR the hip rather than swung out at the
# end of a still-mostly-extended shank.
#
# Unlike rear_stand_rear_leg_extension (which stayed calf-only because thigh's sign
# wasn't independently verified), BOTH joints are anchored here -- the FK derivation
# above was carried out explicitly for this file rather than inherited unverified.
# Still a FIRST GUESS on the exact numbers, same as the rest of this block; if this
# term measurably fights the height/CoM objectives on the bench, thigh is the one to
# revisit first (it carries the geometric assumption, calf's role is simpler).
THIGH_FOLD_TARGET = 2.4  # rad
CALF_FOLD_TARGET = -2.5  # rad

# Direction mapping v8 (owner's spec, see module docstring) -- forward=FR, right=RR,
# back=RL, left=FL. Order below is (dx, dy) per direction, used ONLY to pick which
# canonical unit vector a newly-resampled cycle commands; leg SELECTION itself is
# computed from the live command vector's own sign/dominance in _selected_leg_mask,
# not by table lookup -- the two are guaranteed consistent by construction since
# both encode the same forward/right/back/left semantics.
CMD_DIRECTIONS = ((1.0, 0.0), (0.0, -1.0), (-1.0, 0.0), (0.0, 1.0))  # fwd,right,back,left
CMD_ACTIVE_THRESHOLD = 0.1  # |cmd| below this reads as "no leg selected" (owner's
# spec: "порог |cmd|>0.1") -- matches the bench driver's own STICK_DEADZONE-gated
# ramp start, see b2_leg_lift_driver.py.

# Explicit, ORDERED (FL,FR,RR,RL) name lists -- every per-leg-shaped reward function
# below relies on this exact order matching _selected_leg_mask's own index
# convention. Unchanged from v7 (this ordering is independent of the v8 direction
# mapping change above -- it's just "which array slot is which body", not "which
# command means which leg").
FOOT_BODY_NAMES = ["FL_calf", "FR_calf", "RR_calf", "RL_calf"]
HIP_BODY_NAMES = ["FL_hip", "FR_hip", "RR_hip", "RL_hip"]
LEG_JOINT_NAMES_ORDERED = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
]  # fmt: skip
# Index of each leg's THIGH/CALF slot within a [N,4,3] (leg, joint) view of
# LEG_JOINT_NAMES_ORDERED-gathered joints -- (hip, thigh, calf) = (0, 1, 2).
_THIGH_IDX, _CALF_IDX = 1, 2


class LegLiftCommand(CommandTerm):
    """Cycle: idle (four legs) -> rise -> hold (one leg lifted) -> descend -> resample,
    same clock trick as jump's JumpPulseCommand / rear_stand's RearStandCommand.

    v8: the exposed `command` is now a genuine 2-slot (lin_vel_x, lin_vel_y) vector --
    a fixed unit direction (picked once per cycle, held until descend completes, same
    idiom as v7) scaled by the SAME 0->1->0 ramp v7 called `signal`. `.signal` is kept
    as an attribute (== the vector's own magnitude) purely so every v7-era reward
    function below that reads `term.signal` keeps working unmodified.

    Also owns the v8 Rudin-style per-env height curriculum (`height_target`,
    `_hold_success`, `_hold_entered`) -- see the module docstring's point 4 and
    LIFT_HEIGHT_* constants above for the full mechanism. Bumped in
    `_resample_command`, which IsaacLab's own CommandTerm.reset()/compute() already
    call at every cycle boundary AND every episode reset (see
    isaaclab/managers/command_manager.py) -- exactly the "per-env, success-gated,
    not on a wall-clock schedule" hook Rudin's own recipe wants."""

    cfg: "LegLiftCommandCfg"

    def __init__(self, cfg: "LegLiftCommandCfg", env) -> None:
        super().__init__(cfg, env)
        n = self.num_envs
        self._command = torch.zeros(n, 2, device=self.device)
        self.signal = torch.zeros(n, device=self.device)
        self.direction = torch.zeros(n, 2, device=self.device)
        self.idle_duration = torch.zeros(n, device=self.device)
        self.hold_duration = torch.zeros(n, device=self.device)
        self.cycle_duration = torch.zeros(n, device=self.device)
        self._directions = torch.tensor(cfg.directions, dtype=torch.float, device=self.device)

        # -- v8 height curriculum state --
        self.height_target = torch.full((n,), LIFT_HEIGHT_INIT, device=self.device)
        self._hold_success = torch.ones(n, dtype=torch.bool, device=self.device)
        self._hold_entered = torch.zeros(n, dtype=torch.bool, device=self.device)

        # Cached body indices for the curriculum's own clearance check (FL,FR,RR,RL
        # order, matching FOOT_BODY_NAMES/HIP_BODY_NAMES) -- resolved once here
        # instead of going through SceneEntityCfg machinery, since this is purely
        # internal bookkeeping, not a manager-registered term.
        body_names = self._env.scene["robot"].body_names
        self._foot_ids = [body_names.index(n) for n in FOOT_BODY_NAMES]
        self._hip_ids = [body_names.index(n) for n in HIP_BODY_NAMES]

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

        # -- v8 height curriculum: judge the cycle that just ended BEFORE resetting
        # the trackers for the new one. `_hold_entered` guards envs on their very
        # first-ever resample (no hold phase has happened yet, nothing to judge).
        # env_ids is always an index tensor here in practice (IsaacLab's top-level
        # env reset resolves None to an explicit torch.arange before it ever reaches
        # a command term -- the only other caller, compute()'s own resample_env_ids,
        # is already a nonzero()-derived tensor), same assumption every other
        # env_ids-indexed line in this class already makes.
        judge = self._hold_entered[env_ids]
        succeeded = judge & self._hold_success[env_ids]
        failed = judge & ~self._hold_success[env_ids]
        up_ids = env_ids[succeeded]
        down_ids = env_ids[failed]
        if len(up_ids) > 0:
            self.height_target[up_ids] = (self.height_target[up_ids] + LIFT_HEIGHT_LEVEL_STEP).clamp(max=LIFT_HEIGHT_MAX)
        if len(down_ids) > 0:
            self.height_target[down_ids] = (self.height_target[down_ids] - LIFT_HEIGHT_LEVEL_DOWN).clamp(min=LIFT_HEIGHT_MIN)
        self._hold_success[env_ids] = True
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

        # -- v8 height curriculum: accumulate whole-hold success while actually
        # holding. Uses the SAME clearance definition as leg_lift_selected_height
        # (selected foot z minus the OTHER three's average z) so the curriculum
        # judges the exact quantity the reward is tracking.
        in_hold = (elapsed >= hold_start) & (elapsed < descend_start)
        if torch.any(in_hold):
            asset = self._env.scene["robot"]
            foot_z = asset.data.body_pos_w[:, self._foot_ids, 2]
            mask = _direction_to_leg_mask(self._command)
            selected_z = (foot_z * mask).sum(dim=1)
            support_z = (foot_z * (1.0 - mask)).sum(dim=1) / 3.0
            clearance = selected_z - support_z
            within_tol = (torch.abs(clearance - self.height_target) < LIFT_HEIGHT_SUCCESS_TOL)
            self._hold_success = torch.where(in_hold, self._hold_success & within_tol, self._hold_success)
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
    by dominant-axis + sign -- v8's own rule (owner's spec: "выбор ноги по квадранту/
    знаку доминирующей оси, порог |cmd|>0.1"), replacing v7's dot-product-against-a-
    direction-table lookup (that table doesn't exist for the new interface -- the
    magnitude IS the signal now, there is no separate unit-direction slot to compare
    against). Below CMD_ACTIVE_THRESHOLD the tie resolves to index 0 (FL) -- same
    harmless default as v7 (every caller also gates by `signal`/magnitude, which is
    ~0 exactly when this tie can fire).

    Mapping (owner's v8 spec): +x(fwd)->FR(1), -y(right)->RR(2), -x(back)->RL(3),
    +y(left)->FL(0)."""
    vx, vy = command[:, 0], command[:, 1]
    x_dominant = torch.abs(vx) >= torch.abs(vy)
    active = torch.linalg.norm(command, dim=-1) > CMD_ACTIVE_THRESHOLD
    idx = torch.zeros(command.shape[0], dtype=torch.long, device=command.device)  # default FL
    idx = torch.where(x_dominant & (vx > 0), torch.full_like(idx, 1), idx)  # forward -> FR
    idx = torch.where(x_dominant & (vx <= 0), torch.full_like(idx, 3), idx)  # back -> RL
    idx = torch.where(~x_dominant & (vy <= 0), torch.full_like(idx, 2), idx)  # right -> RR
    idx = torch.where(~x_dominant & (vy > 0), torch.full_like(idx, 0), idx)  # left -> FL
    idx = torch.where(active, idx, torch.zeros_like(idx))
    mask = torch.zeros(command.shape[0], 4, device=command.device, dtype=command.dtype)
    mask.scatter_(1, idx.unsqueeze(-1), 1.0)
    return mask


def _selected_leg_mask(env, command_name: str) -> torch.Tensor:
    """Reward-function-facing wrapper around _direction_to_leg_mask -- fetches the
    live command tensor from the named command term. Kept as a separate function
    (v7 also had one under this exact name/signature) so every reward function below
    that already calls `_selected_leg_mask(env, command_name)` needed zero changes."""
    term = env.command_manager.get_term(command_name)
    return _direction_to_leg_mask(term.command)


def leg_lift_selected_height(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    hip_cfg: SceneEntityCfg,
    xy_tolerance: float,
) -> torch.Tensor:
    """Track the COMMANDED leg's foot clearance above the OTHER THREE's own average
    height -- self-calibrating (relative to the live support tripod), unchanged
    mechanism from v7. v8 change: the target is no longer a fixed constant -- it
    reads `term.height_target`, the Rudin-style PER-ENV curriculum value (see
    LegLiftCommand's own docstring), ramped by the same signal as before (0 at idle,
    height_target once fully commanded).

    GATED by horizontal proximity to the hip (v7 fix, kept verbatim -- see git
    history for the original diagnosis: height credit requires the foot to already
    be near the hip, so sweeping the leg back earns ~0 on both this and
    leg_lift_foot_horizontal at once)."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    foot_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]  # [N,4], FL,FR,RR,RL order
    mask = _selected_leg_mask(env, command_name)
    selected_z = (foot_z * mask).sum(dim=1)
    support_z = (foot_z * (1.0 - mask)).sum(dim=1) / 3.0
    clearance = selected_z - support_z
    target = term.height_target * term.signal
    height_kernel = torch.exp(-torch.square(clearance - target) / TRACKING_SIGMA)

    foot_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, 0:2]
    hip_xy = asset.data.body_pos_w[:, hip_cfg.body_ids, 0:2]
    dist = torch.linalg.norm(foot_xy - hip_xy, dim=-1)  # [N,4]
    selected_dist = (dist * mask).sum(dim=1)
    excess = (selected_dist - xy_tolerance).clamp(min=0.0)
    proximity_gate = torch.exp(-torch.square(excess) / LIFT_XY_GATE_SIGMA)

    return height_kernel * proximity_gate


def leg_lift_support_contact(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize the THREE support feet losing contact while a lift is actually
    commanded. Unchanged from v7 -- only depends on `_selected_leg_mask`/`term.signal`,
    both still present under the v8 interface."""
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
    tolerance band. Unchanged from v7 -- see leg_lift_selected_height's own docstring
    for why this pairs with the proximity gate as belt-and-suspenders."""
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
    """L2 base-height anchor active while a lift is commanded. Unchanged from v7."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    err = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return err * term.signal


def leg_lift_base_still(env, command_name: str) -> torch.Tensor:
    """Penalty on horizontal base velocity + yaw rate, active in EVERY phase.
    Unchanged from v7."""
    asset = env.scene["robot"]
    v_xy = torch.sum(torch.square(asset.data.root_lin_vel_w[:, 0:2]), dim=1)
    w_z = torch.square(asset.data.root_ang_vel_w[:, 2])
    return v_xy + w_z


def leg_lift_support_pose(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """L1 penalty on joint deviation from default, masked to the THREE support legs
    only, exemption scaled by the lift signal (v4-era fix, kept). Unchanged from v7
    -- see git history for the full idle-drift/"кульбит" diagnosis this fixed."""
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


def leg_lift_com_over_support(env, command_name: str, foot_cfg: SceneEntityCfg) -> torch.Tensor:
    """v8 NEW. Base XY centered over the CENTROID of the three SUPPORT feet (not a
    fixed pair, and not all four) -- the fault-tolerant-locomotion mechanism the task
    doc points at (arXiv:2606.25965, MoE RL/IsaacLab/Go2): with one leg lifted, three
    feet is the entire base of support, and the CoM belongs inside that triangle.
    Sibling of rear_stand_com_over_support (same exp(-8*err) kernel, same reasoning),
    generalized from a 2-foot midpoint to a 3-foot centroid via the same per-env
    selection mask every other function in this file already uses. Gated on the lift
    signal actually being underway (idle = 4-leg support, the CoM belongs centered
    over all four, not one particular triangle)."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    foot_xy = asset.data.body_pos_w[:, foot_cfg.body_ids, 0:2]  # [N,4,2] FL,FR,RR,RL
    mask = _selected_leg_mask(env, command_name)
    support_centroid = (foot_xy * (1.0 - mask).unsqueeze(-1)).sum(dim=1) / 3.0
    base_xy = asset.data.root_pos_w[:, 0:2]
    err = torch.sum(torch.square(base_xy - support_centroid), dim=1)
    return torch.exp(-8.0 * err) * term.signal


def leg_lift_feet_on_air(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """v8 NEW. Positive one-shot bonus the instant the SELECTED leg leaves the
    ground, ported from unitree_a1_handstand's own handstand_feet_on_air. Their
    version uses a static body subset (torch.all across a fixed set of feet meant to
    be airborne); here the "which foot" varies per env, so first_air is computed for
    all four and reduced through the same per-env selection mask as every other
    function in this file, instead of torch.all over a fixed subset.

    Gated on signal>0.5 (same threshold rear_stand_front_feet_on_air uses) rather
    than any nonzero signal, so a brief noise-triggered liftoff during the rise ramp
    isn't credited the same as a genuine held lift."""
    term = env.command_manager.get_term(command_name)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids].float()  # [N,4]
    mask = _selected_leg_mask(env, command_name)
    reward = (first_air * mask).sum(dim=1)
    risen = (term.signal > 0.5).float()
    return reward * risen


def leg_lift_feet_air_time(env, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """v8 NEW. Rewards sustained air time on the SELECTED leg past `threshold`,
    ported from unitree_a1_handstand's own handstand_feet_air_time, same per-env
    masking as leg_lift_feet_on_air above (their static-subset torch.sum over a
    fixed set of feet becomes a masked sum over whichever leg this env's command
    actually selected)."""
    term = env.command_manager.get_term(command_name)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward_all = (last_air_time - threshold) * first_contact.float()
    mask = _selected_leg_mask(env, command_name)
    return (reward_all * mask).sum(dim=1)


def leg_lift_joint_fold(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """v8 NEW. Secondary joint-angle anchor: exp-tracking of the SELECTED leg's
    thigh+calf toward (THIGH_FOLD_TARGET, CALF_FOLD_TARGET) -- "foot folded up near
    the body", decoupled from the pure world-Z clearance leg_lift_selected_height
    already owns, same "two co-objectives, not one proxy for the other" structure as
    rear_stand_rear_leg_extension alongside rear_stand_front_feet_height. Ramped by
    the same signal (default pose at idle, fold target once fully commanded), masked
    to the selected leg only via the same per-env mask as every other function here
    -- the three support legs' own thigh/calf stay governed by leg_lift_support_pose
    instead, so there is no double-anchoring."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids].view(-1, 4, 3)  # [N,leg,joint]
    default = asset.data.default_joint_pos[:, asset_cfg.joint_ids].view(-1, 4, 3)
    fold_target = default.clone()
    fold_target[:, :, _THIGH_IDX] = THIGH_FOLD_TARGET
    fold_target[:, :, _CALF_IDX] = CALF_FOLD_TARGET
    signal = term.signal.view(-1, 1, 1)
    target = default + (fold_target - default) * signal
    err = torch.sum(
        torch.square(joint_pos[:, :, _THIGH_IDX:_CALF_IDX + 1] - target[:, :, _THIGH_IDX:_CALF_IDX + 1]), dim=2
    )  # [N,4]
    kernel = torch.exp(-err / TRACKING_SIGMA)
    mask = _selected_leg_mask(env, command_name)
    return (kernel * mask).sum(dim=1)


class UnitreeB2LegLiftRoughEnvCfg(UnitreeB2RoughEnvCfg):
    """See module docstring -- a deliberately simple test skill, not a production one."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Flat ground -- unchanged reasoning from v7: the trick is plenty hard
        # without rough terrain on top.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        # The lift command replaces the velocity command wholesale. v8: now 2 slots
        # (lin_vel_x, lin_vel_y), not v7's 3 -- see module docstring.
        self.commands.base_velocity = LegLiftCommandCfg()
        self.curriculum.command_levels = None

        # -- retire the velocity-tracking recipe, same as every other non-gait B2 variant
        self.rewards.track_lin_vel_xy_exp.weight = 0
        self.rewards.track_ang_vel_z_exp.weight = 0

        # Not a periodic gait -- feet_height/feet_height_body retired in favor of
        # leg_lift_selected_height, which is specific to the one commanded leg.
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height_body.weight = 0

        # Single-leg lift is inherently ASYMMETRIC -- joint_mirror would directly
        # fight the entire point of this task.
        self.rewards.joint_mirror.weight = 0

        # See leg_lift_support_pose's own docstring for why these two generic terms
        # are retired here rather than left active alongside it.
        self.rewards.stand_still_without_cmd.weight = 0
        self.rewards.joint_pos_penalty.weight = 0

        # v7 values, kept as-is for v8 -- "ровный корпус ВСЕГДА" (owner's rule) plus
        # the feet_slide fix for the idle-tilt/"кульбит" and dragging-instead-of-
        # lifting symptoms; both proven on the v2/v7 lineage, not touched by this
        # round's redesign (which targets the missing POSITIVE mechanisms, not this
        # pair).
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.feet_slide.weight = -0.5

        # -- the lift objective itself (v7 mechanism, v8 per-env curriculum target)
        self.rewards.leg_lift_selected_height = RewTerm(
            func=leg_lift_selected_height,
            weight=6.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True),
                "hip_cfg": SceneEntityCfg("robot", body_names=HIP_BODY_NAMES, preserve_order=True),
                "xy_tolerance": LIFT_XY_TOLERANCE,
            },
        )
        self.rewards.leg_lift_support_contact = RewTerm(
            func=leg_lift_support_contact,
            weight=-3.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
            },
        )
        self.rewards.leg_lift_base_still = RewTerm(
            func=leg_lift_base_still,
            weight=-2.0,
            params={"command_name": "base_velocity"},
        )
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
        self.rewards.leg_lift_base_height = RewTerm(
            func=leg_lift_base_height,
            weight=-8.0,
            params={"command_name": "base_velocity", "target_height": LIFT_BASE_HEIGHT_TARGET},
        )
        self.rewards.leg_lift_support_pose = RewTerm(
            func=leg_lift_support_pose,
            weight=-4.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES_ORDERED, preserve_order=True),
            },
        )

        # -- v8 NEW terms --
        # Fault-tolerant CoM-over-support-triangle: the main balance mechanism this
        # task needs (task doc: "главный и почти единственный балансный терм"),
        # weighted just under the height objective itself (6.0) since it is a
        # co-primary requirement, not a minor backstop.
        self.rewards.leg_lift_com_over_support = RewTerm(
            func=leg_lift_com_over_support,
            weight=5.0,
            params={
                "command_name": "base_velocity",
                "foot_cfg": SceneEntityCfg("robot", body_names=FOOT_BODY_NAMES, preserve_order=True),
            },
        )
        # Secondary joint-fold anchor -- see leg_lift_joint_fold's own docstring.
        # Weighted below the two co-primary terms (height 6.0, CoM 5.0) since it is
        # explicitly a secondary/decoupling anchor, same relative positioning
        # rear_stand_rear_leg_extension (6.0) had under rear_stand_front_feet_height
        # (10.0)/rear_stand_com_over_support (2.0) in that file.
        self.rewards.leg_lift_joint_fold = RewTerm(
            func=leg_lift_joint_fold,
            weight=4.0,
            params={
                "command_name": "base_velocity",
                "asset_cfg": SceneEntityCfg("robot", joint_names=LEG_JOINT_NAMES_ORDERED, preserve_order=True),
            },
        )
        # Positive air package -- task doc calls these "дешёвые термы" (cheap terms):
        # they price the FACT/DURATION of the leg actually leaving the ground, which
        # nothing else here does (selected_height only prices clearance once
        # airborne, not the transition itself). Weighted well under the co-primary
        # pair, same relative scale rear_stand's own front_feet_on_air/air_time
        # (5.0/5.0) had under its dominant front_feet_height (10.0).
        self.rewards.leg_lift_feet_on_air = RewTerm(
            func=leg_lift_feet_on_air,
            weight=3.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
            },
        )
        self.rewards.leg_lift_feet_air_time = RewTerm(
            func=leg_lift_feet_air_time,
            weight=3.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES, preserve_order=True),
                # Hold windows are 1.5-3.0s (HOLD_TIME_RANGE) -- 1.0s threshold asks
                # for a genuinely SUSTAINED lift (not just a touch-and-go), same
                # "scaled to this task's own hold window" reasoning
                # rear_stand_front_feet_air_time's own threshold=3.0 used against
                # rear_stand's much longer 8-14s hold_time_range. FIRST GUESS.
                "threshold": 1.0,
            },
        )

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeB2LegLiftRoughEnvCfg":
            self.disable_zero_weight_rewards()
