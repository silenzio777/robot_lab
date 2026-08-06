# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Rear-leg stand ("свечка") for B2 -- port of workshop_legged_gym's go2_stand task
(MIPT ROS-meetup fork of ETH legged_gym, found by the user 2026-08-05) into this
repo's IsaacLab manager-based stack.

What the original does (legged_gym/envs/go2_stand/): a Go2 learns to rear up into a
vertical stand on its hind legs via
  - tracking_pitch (w=5): the pitch TARGET ramps linearly 0 -> -90deg over
    standup_duration seconds of episode time -- a choreographed transition reference,
    not "figure out the motion yourself";
  - base_height (w=3): exp-tracking of the standing-tall base height;
  - com_over_support: exp reward for keeping the base centered over the midpoint of
    the two REAR feet (the bipedal support polygon);
  - rear_feet_contact_and_air (w=4): rear feet must touch, front feet must NOT
    (undesired-contact penalty -15 inside the term);
  - the usual regularizers + heavy domain rand.

Port notes (not a copy -- different framework, different robot):
- Orientation is tracked via the world-Z component of the BODY's forward (+X) axis
  (0 when horizontal -> 1 when vertical), not euler pitch: same quantity, no gimbal
  trouble exactly at the -90deg target. Roll is held via the body +Y axis staying
  horizontal.
- All recipe terms that pull toward the DEFAULT quadruped stand are zeroed here:
  upward (rewards flat orientation!), stand_still_without_cmd / joint_pos_penalty
  (pull joints to the four-legged default pose), feet_height_body (feet-below-body
  geometry breaks when vertical), feet_contact_without_cmd (front feet must be OFF
  the ground -- the opposite), lin_vel_z_l2 / ang_vel_xy_l2 (the rise itself is a
  pitch rotation with vertical motion).
- Flat plane terrain; height_scanner + critic scan kept alive for checkpoint-shape
  compatibility (same lesson as jump_env_cfg's own warm-start note).
- B2 numbers are first guesses off its geometry (body ~1.1m, rear-leg stand puts
  base_link around 0.85m) -- calibrate from what training produces, same as every
  other variant's own first pass.

Velocity commands are zeroed (pure stand, v1). The original could WALK bipedally
(lin_vel_x +-0.3 while vertical) -- that's a natural v2 once the stand holds.
"""

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeB2RoughEnvCfg

# The choreographed rise: target for the body-forward-axis world-Z component ramps
# 0 -> 1 over this many seconds of episode time (go2_stand used 1.25s for a much
# lighter robot; B2 is ~60kg, give it a bit longer).
# 1.5 -> 2.0 (2026-08-06, first-run plateau): a gentler rise trajectory -- the
# policy settled into a half-rear "beg" pose ~30deg short of vertical.
STANDUP_DURATION = 2.0
# Base height once vertical -- rear-leg stand geometry first guess, calibrate.
STAND_HEIGHT_TARGET = 0.85
# 0.5 -> 0.25 (2026-08-06, same plateau): with the original broad kernel the
# gradient near the top is flat -- at ~66% tracking the marginal reward for the
# last 30deg didn't beat the fall risk, so the policy parked (reward flat 1100+
# iterations, noise converged to 0.74). Sharper sigma makes the shortfall
# expensive exactly where the first run stalled.
TRACKING_SIGMA = 0.25


# v2 (2026-08-06, after the first bench test of the v1 stand): the rise is now
# COMMAND-driven, not episode-time-driven, and the cycle includes a trained DESCENT.
# Bench findings that forced this: (a) v1's permanent-stand policy had no exit --
# switching to walk from the risen pose tangled the legs and fell; (b) with stance
# width unpriced, the policy splayed the rear legs into a statically-stable but
# walk-incompatible straddle (the free-variable lesson, again). The command signal
# in slot 0 leads the policy through four-legs -> rise -> hold -> descend -> four-
# legs; slot 1 is reserved for the future bipedal walking command (v3).
RISE_DURATION = 2.0
DESCEND_DURATION = 2.0
# Natural rear-feet lateral separation, measured on the model's default stance --
# bipedal walking needs the feet under the hips, not a straddle.
STANCE_WIDTH_TARGET = 0.35

# v3 (2026-08-06, v2 cycle confirmed on the bench): BIPEDAL WALKING while vertical.
# Command slot 1 (reserved since v2) now carries a walk velocity, active only during
# the hold phase: vx > 0 = step toward where the belly faces (the horizontal
# projection of the body -Z axis -- when nose-up that is exactly the original
# heading), vx < 0 = backward. MIPT's go2_stand walked with lin_vel_x in
# [-0.3, 0.3] while tracking pitch; same range here.
WALK_VX_RANGE = (-0.3, 0.3)
WALK_ZERO_PROB = 0.3  # a share of cycles hold still -- pure standing must survive v3
WALK_RAMP = 0.3  # s, smooth on/off of the walk command inside the hold window
WALK_START_DELAY = 0.5  # s into hold before the walk command switches on


class RearStandCommand(CommandTerm):
    """Cycle: idle (four legs) -> rise -> hold vertical -> descend -> resample.
    Same clock trick as jump's JumpPulseCommand: _resample_command overwrites the
    base class's time_left with the full cycle length."""

    cfg: "RearStandCommandCfg"

    def __init__(self, cfg: "RearStandCommandCfg", env) -> None:
        super().__init__(cfg, env)
        n = self.num_envs
        self._command = torch.zeros(n, 3, device=self.device)
        self.signal = torch.zeros(n, device=self.device)
        self.idle_duration = torch.zeros(n, device=self.device)
        self.hold_duration = torch.zeros(n, device=self.device)
        self.cycle_duration = torch.zeros(n, device=self.device)
        # v3: per-cycle bipedal walk velocity (slot 1, hold phase only).
        self.walk_vx = torch.zeros(n, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        self.metrics["stand_signal"] = self.signal.clone()

    def _resample_command(self, env_ids):
        idle = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.idle_time_range)
        hold = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.hold_time_range)
        self.idle_duration[env_ids] = idle
        self.hold_duration[env_ids] = hold
        self.cycle_duration[env_ids] = idle + RISE_DURATION + hold + DESCEND_DURATION
        self.time_left[env_ids] = self.cycle_duration[env_ids]
        # v3: walk velocity for this cycle's hold phase; a share of cycles stay
        # at 0 so pure standing keeps its own training signal.
        vx = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.walk_vx_range)
        still = torch.rand(len(env_ids), device=self.device) < self.cfg.walk_zero_prob
        self.walk_vx[env_ids] = torch.where(still, torch.zeros_like(vx), vx)

    def _update_command(self):
        elapsed = self.cycle_duration - self.time_left
        rise_start = self.idle_duration
        hold_start = rise_start + RISE_DURATION
        descend_start = hold_start + self.hold_duration
        rising = ((elapsed - rise_start) / RISE_DURATION).clamp(0.0, 1.0)
        descending = ((elapsed - descend_start) / DESCEND_DURATION).clamp(0.0, 1.0)
        self.signal = rising - descending  # 0 idle, ramps up, holds 1, ramps down
        self._command[:, 0] = self.signal
        # v3: slot 1 = walk vx, smoothly ramped on after WALK_START_DELAY into the
        # hold and back off before the descend begins.
        walk_on = hold_start + WALK_START_DELAY
        ramp_in = ((elapsed - walk_on) / WALK_RAMP).clamp(0.0, 1.0)
        ramp_out = ((descend_start - elapsed) / WALK_RAMP).clamp(0.0, 1.0)
        self._command[:, 1] = self.walk_vx * ramp_in * ramp_out


@configclass
class RearStandCommandCfg(CommandTermCfg):
    class_type: type = RearStandCommand
    resampling_time_range: tuple[float, float] = (8.0, 14.0)  # nominal; overwritten per cycle
    idle_time_range: tuple[float, float] = (2.0, 4.0)
    # v3: hold extended (3,6)->(4,8) so a walking cycle has room to actually walk.
    hold_time_range: tuple[float, float] = (4.0, 8.0)
    walk_vx_range: tuple[float, float] = WALK_VX_RANGE
    walk_zero_prob: float = WALK_ZERO_PROB
    debug_vis: bool = False


def _fwd_axis_z(asset) -> torch.Tensor:
    """World-Z component of the body +X axis: 0 horizontal, 1 nose-straight-up."""
    q = asset.data.root_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return 2.0 * (x * z - w * y)


def _side_axis_z(asset) -> torch.Tensor:
    """World-Z component of the body +Y axis: 0 when roll-free at any pitch."""
    q = asset.data.root_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return 2.0 * (y * z + w * x)


def _walk_dir_xy(asset) -> torch.Tensor:
    """Unit horizontal direction the BELLY faces = -(body Z axis) projected to the
    ground plane (v3). When the robot is nose-up vertical, the body -Z axis lands
    exactly on the pre-rise heading (torch-checked: q=(0.707,0,-0.707,0) ->
    (1,0,0)), so walk vx > 0 means "step the way you were facing before rearing".
    Degenerate when horizontal (projection -> 0) -- callers gate on being risen."""
    q = asset.data.root_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    # Body +Z axis in world, XY part; belly faces the opposite way.
    dir_xy = torch.stack([-2.0 * (x * z + w * y), -2.0 * (y * z - w * x)], dim=1)
    return dir_xy / dir_xy.norm(dim=1, keepdim=True).clamp(min=1e-6)


def _risen_mask(env, asset) -> torch.Tensor:
    """Commanded-and-actually vertical: the walk terms must pay ONLY for bipedal
    locomotion -- without the measured-verticality gate a policy that ignored the
    rise could farm walk tracking on four legs (the free-variable lesson,
    preempted this time)."""
    commanded = (env.command_manager.get_term("base_velocity").signal > 0.9).float()
    actually = (_fwd_axis_z(asset) > 0.7).float()
    return commanded * actually


def rear_stand_orientation_tracking(env, asset_cfg=None) -> torch.Tensor:
    """Follow the COMMANDED verticality (v2: the command signal IS the target --
    the policy observes it in command slot 0 and must track it up AND down),
    roll held flat -- exp kernel."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    target = env.command_manager.get_term("base_velocity").signal
    err = torch.square(target - _fwd_axis_z(asset)) + torch.square(_side_axis_z(asset))
    return torch.exp(-err / TRACKING_SIGMA)


def rear_stand_height(env, asset_cfg=None) -> torch.Tensor:
    """exp-tracking of the vertical-stand base height (flat plane -> world Z is
    ground-true). Ramped with the same clock as orientation so the two references
    never contradict each other mid-rise."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    signal = env.command_manager.get_term("base_velocity").signal
    # From the quadruped stand (0.55) up to the rear-stand target, led by the command.
    target = 0.55 + (STAND_HEIGHT_TARGET - 0.55) * signal
    err = torch.square(asset.data.root_pos_w[:, 2] - target)
    return torch.exp(-err / TRACKING_SIGMA)


def rear_stand_com_over_support(env, asset_cfg=None) -> torch.Tensor:
    """Base XY centered over the midpoint of the rear feet -- the original's
    com_over_support, height part dropped (rear_stand_height owns height)."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    body_names = asset.body_names
    rl = body_names.index("RL_calf")
    rr = body_names.index("RR_calf")
    base_xy = asset.data.root_pos_w[:, 0:2]
    support_xy = 0.5 * (asset.data.body_pos_w[:, rl, 0:2] + asset.data.body_pos_w[:, rr, 0:2])
    err = torch.sum(torch.square(base_xy - support_xy), dim=1)
    # Only meaningful once actually rearing (v2): on four legs the CoM belongs
    # between all four feet, not over the rear pair.
    risen = (env.command_manager.get_term("base_velocity").signal > 0.5).float()
    return torch.exp(-8.0 * err) * risen


def rear_stand_front_feet_contact(env, sensor_cfg=None) -> torch.Tensor:
    """Count of FRONT feet in contact -- weighted negative: once the rise begins the
    front feet belong in the air (the original folded this into
    rear_feet_contact_and_air with a -15 inner factor). Gated on the ramp actually
    being underway so the pre-rise instant isn't penalized."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = (contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0).float()
    risen = (env.command_manager.get_term("base_velocity").signal > 0.5).float()
    return in_contact.sum(dim=1) * risen


def rear_stand_stance_width(env, asset_cfg=None) -> torch.Tensor:
    """Penalty on rear-feet LATERAL separation deviating from the natural stance
    width while rearing (v2, bench finding: unpriced width let the policy splay
    into a statically-stable straddle that bipedal walking can never use --
    the free-variable lesson, instance N).
    v3 change: penalize only the component PERPENDICULAR to the walk direction --
    the full planar norm would bill every stride (stepping separates the feet
    fore-aft along the walk axis) at w=-8, crushing walking before it starts."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    body_names = asset.body_names
    rl = body_names.index("RL_calf")
    rr = body_names.index("RR_calf")
    d = asset.data.body_pos_w[:, rl, 0:2] - asset.data.body_pos_w[:, rr, 0:2]
    walk_dir = _walk_dir_xy(asset)
    # |cross_z(walk_dir, d)| = separation perpendicular to the walk axis.
    lateral = torch.abs(walk_dir[:, 0] * d[:, 1] - walk_dir[:, 1] * d[:, 0])
    err = torch.square(lateral - STANCE_WIDTH_TARGET)
    risen = (env.command_manager.get_term("base_velocity").signal > 0.5).float()
    return err * risen


def rear_stand_idle_still(env, asset_cfg=None) -> torch.Tensor:
    """Stillness on four legs (command signal ~0): horizontal drift and yaw cost --
    the jump task's own idle lesson, applied from day one here."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    v_xy = torch.sum(torch.square(asset.data.root_lin_vel_w[:, 0:2]), dim=1)
    w_z = torch.square(asset.data.root_ang_vel_w[:, 2])
    idle = (env.command_manager.get_term("base_velocity").signal < 0.1).float()
    return (v_xy + w_z) * idle


def rear_stand_front_feet_down_idle(env, sensor_cfg=None) -> torch.Tensor:
    """Front feet ON the ground while the command says four-legs (signal ~0).
    Added 2026-08-06 after BOTH the resumed and the from-scratch v2 runs parked in
    the same attractor -- a permanent half-rise that ignores the command (identical
    term values from two different starts = it's the task economics, not the warm
    start). Nothing rewarded actually being four-legged in idle: orientation/height
    targets pull that way but a half-rise loses only part of their exp reward,
    and no term made front-feet-down itself profitable. Now idle has its own
    positive signature the half-rise can't collect."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = (contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0).float()
    idle = (env.command_manager.get_term("base_velocity").signal < 0.1).float()
    return in_contact.mean(dim=1) * idle


def rear_stand_rear_feet_contact(env, sensor_cfg=None) -> torch.Tensor:
    """REAR feet are the support. v3 change: while a walk command is active a
    stride NEEDS one foot in the air -- demanding both down (v2's mean) would tax
    every step at w=1. Walking pays for at-least-one-down (max); standing/rising/
    descending still pays for both (mean)."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = (contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0).float()
    walking = (torch.abs(env.command_manager.get_term("base_velocity").command[:, 1]) > 0.05).float()
    return walking * in_contact.max(dim=1).values + (1.0 - walking) * in_contact.mean(dim=1)


def rear_stand_walk_tracking(env, asset_cfg=None) -> torch.Tensor:
    """v3: exp-tracking of the commanded bipedal walk velocity (slot 1) along the
    belly-forward axis. Gated by _risen_mask -- pays only for walking that happens
    UPRIGHT. Note the zero command is tracked too (hold-still cycles), so this
    term also steadies the pure stand instead of fighting it."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    cmd_vx = env.command_manager.get_term("base_velocity").command[:, 1]
    v_walk = torch.sum(asset.data.root_lin_vel_w[:, 0:2] * _walk_dir_xy(asset), dim=1)
    err = torch.square(cmd_vx - v_walk)
    return torch.exp(-err / TRACKING_SIGMA) * _risen_mask(env, asset)


def rear_stand_walk_drift(env, asset_cfg=None) -> torch.Tensor:
    """v3: price the free variables of upright locomotion -- lateral slide
    (perpendicular to the walk axis) and yaw spin. Same preemption as jump's
    direction_precision: nothing else bills sideways drift once risen (idle_still
    is gated to signal<0.1)."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    walk_dir = _walk_dir_xy(asset)
    v_xy = asset.data.root_lin_vel_w[:, 0:2]
    v_perp = v_xy[:, 0] * walk_dir[:, 1] - v_xy[:, 1] * walk_dir[:, 0]
    w_z = torch.square(asset.data.root_ang_vel_w[:, 2])
    return (torch.square(v_perp) + w_z) * _risen_mask(env, asset)


@configclass
class UnitreeB2RearStandRoughEnvCfg(UnitreeB2RoughEnvCfg):
    """See module docstring -- rear-leg vertical stand, ported from go2_stand."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Flat plane; scanner/critic scan kept for checkpoint-shape compatibility
        # (same reasoning as jump_env_cfg's own flat-switch note).
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None
        self.curriculum.command_levels = None

        # v2: the stance-cycle command replaces the velocity command wholesale
        # (same 3 slots -> observation layout unchanged; slot 0 = stand signal,
        # slot 1 reserved for the future bipedal walking command).
        self.commands.base_velocity = RearStandCommandCfg()

        # -- retire everything that pulls toward the flat quadruped stand
        self.rewards.track_lin_vel_xy_exp.weight = 0
        self.rewards.track_ang_vel_z_exp.weight = 0
        self.rewards.upward.weight = 0  # rewards FLAT orientation -- the exact opposite
        self.rewards.stand_still_without_cmd.weight = 0  # pulls to 4-leg default pose
        self.rewards.joint_pos_penalty.weight = 0  # same
        self.rewards.feet_height_body.weight = 0  # feet-below-body breaks when vertical
        self.rewards.feet_contact_without_cmd.weight = 0  # front feet must be OFF
        self.rewards.lin_vel_z_l2.weight = 0  # the rise IS vertical motion
        self.rewards.ang_vel_xy_l2.weight = 0  # the rise IS a pitch rotation

        # -- the rear-stand objective (weights mirror go2_stand's own proportions)
        # 5 -> 8 (2026-08-06, both v2 starts parked half-risen): with no falls at
        # all (episodes flat 1000), the moat around the ignore-the-command optimum
        # is not risk -- it's gradient weakness; make following the signal pay
        # decisively more than parking.
        self.rewards.rear_stand_orientation_tracking = RewTerm(
            func=rear_stand_orientation_tracking, weight=8.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        self.rewards.rear_stand_height = RewTerm(
            func=rear_stand_height, weight=3.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        self.rewards.rear_stand_com_over_support = RewTerm(
            func=rear_stand_com_over_support, weight=2.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        self.rewards.rear_stand_front_feet_contact = RewTerm(
            func=rear_stand_front_feet_contact,
            weight=-2.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"])},
        )
        self.rewards.rear_stand_front_feet_down_idle = RewTerm(
            func=rear_stand_front_feet_down_idle,
            weight=1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"])},
        )
        self.rewards.rear_stand_stance_width = RewTerm(
            func=rear_stand_stance_width, weight=-8.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        self.rewards.rear_stand_idle_still = RewTerm(
            func=rear_stand_idle_still, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        self.rewards.rear_stand_rear_feet_contact = RewTerm(
            func=rear_stand_rear_feet_contact,
            weight=1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"])},
        )

        # -- v3: bipedal walking while vertical (command slot 1, hold phase).
        # Tracking 3.0: enough to make stepping pay, deliberately below
        # orientation's 8 -- falling over to chase velocity must never win.
        self.rewards.rear_stand_walk_tracking = RewTerm(
            func=rear_stand_walk_tracking, weight=3.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        self.rewards.rear_stand_walk_drift = RewTerm(
            func=rear_stand_walk_drift, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeB2RearStandRoughEnvCfg":
            self.disable_zero_weight_rewards()
