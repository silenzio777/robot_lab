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
# 0.85 -> 0.62 (v6.2, 2026-08-17, owner's own doubt confirmed by URDF geometry):
# thigh=0.35m + calf=0.35m (b2.urdf: FL_calf_joint origin z=-0.35 off thigh,
# FL_foot_joint origin z=-0.35 off calf) = 0.70m hip-to-foot MAXIMUM reach at a
# fully straight knee -- a mechanical singularity, never actually reachable (zero
# force capability, no real quadruped/biped stands locked-straight). 0.85 EXCEEDED
# even that impossible ceiling by 15cm. Explains the v6.1 postmortem exactly:
# height climbed genuinely (sigma fix worked) but decelerated approaching a target
# past the robot's own kinematic limit, not "needs more time". 0.62 = ~89% of the
# 0.70m theoretical max, leaving room for the slight functional knee bend a real
# stand needs (a straight-locked leg can't correct for sway). First real value
# this constant has had since its own "first guess, calibrate from what training
# produces" comment (2026-08-06) -- now grounded in the actual mesh geometry.
STAND_HEIGHT_TARGET = 0.62
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

# v4 (2026-08-09, user: "Давай делать" -- re-read workshop_legged_gym's own
# go2_stand/go2.py commit history for the real mechanism behind its bench-confirmed
# rear-leg walk+turn ("very good walk on rear legs on real robot",
# "адекватная ходьба и управление влево или вправо"). Two capabilities ported that
# v3 never had at all:
#
# 1. TURNING (command slot 4): v3 had no yaw channel whatsoever -- the bench driver
#    repurposed the yaw stick as DESCEND purely because nothing else used it. The
#    original trains ang_vel_yaw [-0.2, 0.2] with its own tracking_ang_vel term;
#    ours mirrors that as a per-cycle categorical alternative to walking (see
#    CMD_*_PROB below) rather than a simultaneous free command, matching this
#    file's own established per-cycle-single-command idiom (v3's walk_vx).
# 2. GAIT PHASE CLOCK (command slots 2-3, sin/cos): the original's single biggest
#    structural ingredient, absent from v3 entirely. A continuous, episode-time-
#    based clock (_get_phase/_get_gait_phase in their go2.py) gives the policy an
#    internal metronome to swing feet against, AND drives an alternating-stance
#    reward (rear_stand_rear_feet_contact, below) instead of v3's undifferentiated
#    "at least one foot down" -- the likely root cause of the bench-confirmed
#    near-zero displacement on both 30599 and 50598 (nothing ever demanded an
#    actual alternating STEP, just some contact). Injected into the SAME command
#    vector consumed by the existing velocity_commands obs term (mdp.generated_commands
#    reads the whole vector already, same trick jump's own phase command slot uses)
#    -- no ObservationsCfg change needed, but the vector grows 3->5, so a v4
#    checkpoint's input layer is NOT warm-start-compatible with 30599's.
TURN_WZ_RANGE = (-0.4, 0.4)  # rad/s -- original used +-0.2 under a looser tracking_sigma=0.5; a first guess, calibrate from what training produces (same convention as every other first-guess constant in this file)

# v5 (2026-08-11, bench verdict on the completed v4 final 39099: rises and stands
# CLEAN, walking/turning DO NOT HAPPEN AT ALL -- walk_tracking 0.16 / turn_tracking
# 0.10 / rear_feet_contact 0.18 in the final training metrics confirm the skills
# never trained, not just failed to transfer. Full redesign of the walking
# economics, four coupled causes diagnosed:
#   1. WALKING EXPOSURE DILUTED: hold was 4-8s inside a ~12-16s cycle, minus
#      WALK_START_DELAY and ramps -- actual commanded-walking time was ~25% of an
#      episode at best. Longer holds + shorter idles below.
#   2. GAIT CLOCK PHYSICALLY TOO FAST: 0.25s/cycle = 4Hz stepping, copied verbatim
#      from the 15kg Go2. A 74.5kg B2 balancing on two feet cannot plausibly cycle
#      its stance at 4Hz -- the schedule was unmatchable, so the gait-contact
#      reward was unearnable and stand-still collected its baseline instead.
#   3. NEAR-ZERO COMMANDS: walk_vx uniform in [-0.3,0.3] makes half the walk
#      cycles ask for |vx| < 0.15, where standing still already collects ~70-95%
#      of the tracking kernel -- the exp-kernel plateau the v4 low_speed term was
#      supposed to fix, but at weight 0.5 it never outbid the risk.
#   4. STEPPING PAID TOO LITTLE: the entire gait mechanism (contact schedule 1.0,
#      clearance 1.0, low_speed 0.5) totaled ~2.5 max against orientation's 8 --
#      committing to a stride risks the 8 to chase the 2.5. go2_stand's own
#      contact-schedule term carries w=4 -- the dominant lever there, an
#      afterthought here. Rebalanced below.
# v6 "stand-only" (2026-08-16, owner's direct staging order after re-watching
# the Go2 reference video: "Нужно делать по частям! Сначала встаем на задние" --
# the reference dog FIRST truly stands up on its rear FEET, torso vertical,
# lower legs half-bent, pressing the feet into the floor; ours never once even
# attempted that in three full runs, it sits on its folded calves instead).
# Phase 1 trains ONLY the rise -> hold-still -> descend cycle: no walking, no
# turning -- those come back as later phases once the true stand exists.
# 0.2 -> 1.0 / 0.5 -> 0.0: every cycle is a pure stand cycle. Command vector
# stays 6-wide (walk/turn slots just always 0, gait clock ticks harmlessly),
# so a phase-1 checkpoint warm-starts phase 2 (walking) without an obs change.
CMD_STILL_PROB = 1.0
CMD_WALK_PROB = 0.0
CMD_TURN_PROB = 1.0 - CMD_STILL_PROB - CMD_WALK_PROB
WALK_VX_MIN = 0.15  # v5: commands below this train nothing (see cause 3 above)
# v5.1 (2026-08-12, user: "можно сделать частоту рулилкой на стенде?"): the
# clock was a CONSTANT (0.5s) through v5 -- the bench could only extrapolate a
# different frequency out-of-distribution (a slider that fakes a metronome the
# network never trained against, no guarantee it responds sanely). Made it a
# genuinely TRAINED control instead, same recipe as walk_vx/turn_wz: sample a
# fresh cadence PER EPISODE from this range, tell the network the actual value
# (new command slot 5, see RearStandCommand.reset), and let
# rear_stand_rear_feet_contact's own existing sin-phase mask do the rest --
# it already prices matching footfall to whatever the current sin/cos says,
# so a faster-commanded cadence mechanically demands faster stepping without
# any reward change. 0.35-0.8s covers roughly 1.25-2.9 Hz -- centered on v5's
# own 0.5s (2Hz), bounded so the fastest case still asks for something a
# 74.5kg robot can plausibly hit (see the v5 module comment, cause 2).
GAIT_CYCLE_RANGE = (0.35, 0.8)  # s, sampled once per episode -- see command slot 5
GAIT_BIAS = 0.2  # double-support tolerance band around each sin zero-crossing, matches go2_stand's own bias
FEET_CLEARANCE_TARGET = 0.05  # m -- matches go2_stand's own target_foot_height; uncalibrated first guess, see rear_stand_feet_clearance's own docstring


class RearStandCommand(CommandTerm):
    """Cycle: idle (four legs) -> rise -> hold vertical -> descend -> resample.
    Same clock trick as jump's JumpPulseCommand: _resample_command overwrites the
    base class's time_left with the full cycle length."""

    cfg: "RearStandCommandCfg"

    def __init__(self, cfg: "RearStandCommandCfg", env) -> None:
        super().__init__(cfg, env)
        n = self.num_envs
        # v4: slots 0=stand signal, 1=walk vx, 2=sin(gait phase), 3=cos(gait phase),
        # 4=turn wz. Was 3 slots through v3 -- NOT warm-start-compatible with any
        # earlier checkpoint (input layer shape changed).
        # v5.1: slot 5=gait_cycle_time (seconds, see GAIT_CYCLE_RANGE) -- grew
        # 5->6, again not warm-start-compatible with v4/v5 checkpoints.
        self._command = torch.zeros(n, 6, device=self.device)
        self.signal = torch.zeros(n, device=self.device)
        self.idle_duration = torch.zeros(n, device=self.device)
        self.hold_duration = torch.zeros(n, device=self.device)
        self.cycle_duration = torch.zeros(n, device=self.device)
        # v3: per-cycle bipedal walk velocity (slot 1, hold phase only).
        self.walk_vx = torch.zeros(n, device=self.device)
        # v4: per-cycle turn rate (slot 4, hold phase only) -- mutually exclusive
        # with walk_vx per cycle (see _resample_command), same reasoning as v3's
        # own single-scalar-command idiom: a combined walk+turn curriculum is a
        # much harder learning problem than two separate skills, revisit once both
        # are individually solid.
        self.turn_wz = torch.zeros(n, device=self.device)
        # v5.1: this episode's gait cadence -- sampled ONCE per true episode
        # reset (see reset() below), NOT per stand-cycle like walk_vx/turn_wz.
        # A robot's stepping tempo doesn't randomly change mid-walk in real
        # life; it's a dial you set and it holds -- exactly how the bench's
        # own "Gait clock (s)" slider is meant to be used.
        self.gait_cycle_time = torch.full((n,), sum(cfg.gait_cycle_range) / 2.0, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        self.metrics["stand_signal"] = self.signal.clone()
        self.metrics["gait_cycle_time"] = self.gait_cycle_time.clone()

    def reset(self, env_ids=None):
        # v5.1: sample this episode's cadence HERE, not in _resample_command --
        # that function also fires mid-episode at every natural stand-cycle
        # boundary (see this class's own docstring, "same clock trick as
        # jump's JumpPulseCommand"), and the cadence must NOT jump around
        # inside one episode (see gait_cycle_time's own comment in __init__).
        # reset() is called only at true episode start (base CommandTerm's own
        # contract), so this is the one place that fires exactly once per life.
        ids = slice(None) if env_ids is None else env_ids
        n = self.gait_cycle_time[ids].shape[0]
        self.gait_cycle_time[ids] = torch.empty(n, device=self.device).uniform_(*self.cfg.gait_cycle_range)
        return super().reset(env_ids)

    def _resample_command(self, env_ids):
        idle = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.idle_time_range)
        hold = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.hold_time_range)
        self.idle_duration[env_ids] = idle
        self.hold_duration[env_ids] = hold
        self.cycle_duration[env_ids] = idle + RISE_DURATION + hold + DESCEND_DURATION
        self.time_left[env_ids] = self.cycle_duration[env_ids]
        # v4: this cycle's hold-phase mode is one of {still, walk, turn} -- a
        # 3-way categorical replacing v3's binary walk_zero_prob so all three
        # skills keep their own clean training signal instead of fighting over a
        # combined command.
        # v5: sample walk speed by MAGNITUDE + random sign instead of uniform over
        # the full signed range -- uniform sampling made half the walk cycles ask
        # for |vx| < 0.15, which standing still already nearly satisfies (see the
        # v5 module comment, cause 3). Every walk cycle now demands a real stride.
        vx_mag = torch.empty(len(env_ids), device=self.device).uniform_(WALK_VX_MIN, self.cfg.walk_vx_range[1])
        vx_sign = torch.where(
            torch.rand(len(env_ids), device=self.device) < 0.5,
            torch.ones(len(env_ids), device=self.device),
            -torch.ones(len(env_ids), device=self.device),
        )
        vx = vx_mag * vx_sign
        wz = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.turn_wz_range)
        mode = torch.rand(len(env_ids), device=self.device)
        is_walk = (mode >= self.cfg.cmd_still_prob) & (mode < self.cfg.cmd_still_prob + self.cfg.cmd_walk_prob)
        is_turn = mode >= self.cfg.cmd_still_prob + self.cfg.cmd_walk_prob
        self.walk_vx[env_ids] = torch.where(is_walk, vx, torch.zeros_like(vx))
        self.turn_wz[env_ids] = torch.where(is_turn, wz, torch.zeros_like(wz))

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
        # hold and back off before the descend begins. v4: slot 4 = turn wz, same
        # ramp (shares walk's own on/off timing -- only one of the two is ever
        # nonzero per cycle, see _resample_command).
        walk_on = hold_start + WALK_START_DELAY
        ramp_in = ((elapsed - walk_on) / WALK_RAMP).clamp(0.0, 1.0)
        ramp_out = ((descend_start - elapsed) / WALK_RAMP).clamp(0.0, 1.0)
        self._command[:, 1] = self.walk_vx * ramp_in * ramp_out
        self._command[:, 4] = self.turn_wz * ramp_in * ramp_out
        # v4: gait phase clock runs continuously off real episode time (go2_stand's
        # own _get_phase), independent of this command's own idle/rise/hold/descend
        # cycle -- always present in the observation so the policy has a metronome
        # available even before a walk cycle starts.
        # v5.1: divides by this EPISODE's own sampled cadence (per-env tensor) instead
        # of a shared constant -- one consistent cadence for the robot's whole life
        # (see gait_cycle_time's own comment), math otherwise unchanged.
        phase = self._env.episode_length_buf.float() * self._env.step_dt / self.gait_cycle_time
        self._command[:, 2] = torch.sin(2.0 * torch.pi * phase)
        self._command[:, 3] = torch.cos(2.0 * torch.pi * phase)
        # v5.1: the cadence VALUE itself, in seconds -- sin/cos alone can't reveal
        # how fast the clock is ticking from a single instantaneous reading (same
        # ambiguity as glancing at a clock face once and not knowing if the hands
        # are fast or slow), so the network needs it as an explicit number to
        # actually condition its stepping speed on. Raw seconds (not Hz, not
        # normalized) to match the bench's own "Gait clock (s)" slider 1:1.
        self._command[:, 5] = self.gait_cycle_time


@configclass
class RearStandCommandCfg(CommandTermCfg):
    class_type: type = RearStandCommand
    resampling_time_range: tuple[float, float] = (8.0, 14.0)  # nominal; overwritten per cycle
    # v5: idle (2,4)->(1.5,3), hold (4,8)->(8,14) -- raise the share of episode
    # time actually spent vertical-with-a-live-command from ~25% to ~60% (see the
    # v5 module comment, cause 1: walking exposure was diluted by the cycle).
    idle_time_range: tuple[float, float] = (1.5, 3.0)
    hold_time_range: tuple[float, float] = (8.0, 14.0)
    walk_vx_range: tuple[float, float] = WALK_VX_RANGE
    walk_zero_prob: float = WALK_ZERO_PROB
    # v4 additions -- see the module-level TURN_WZ_RANGE/CMD_*_PROB comment above.
    turn_wz_range: tuple[float, float] = TURN_WZ_RANGE
    cmd_still_prob: float = CMD_STILL_PROB
    cmd_walk_prob: float = CMD_WALK_PROB
    # v5.1 addition -- see GAIT_CYCLE_RANGE's own module-level comment.
    gait_cycle_range: tuple[float, float] = GAIT_CYCLE_RANGE
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


def _gait_mask(env) -> torch.Tensor:
    """[num_envs, 2] bool (RR, RL) -- which rear foot should be in STANCE right
    now, per the command's own sin-phase clock (slot 2) -- go2_stand's own
    _get_gait_phase, ported: RR stances the first half-cycle, RL the second, both
    stance through a short double-support window (|sin| < GAIT_BIAS) around each
    crossing so the swing leg has time to land before its partner lifts."""
    sin_pos = env.command_manager.get_term("base_velocity").command[:, 2]
    mask = torch.zeros(sin_pos.shape[0], 2, dtype=torch.bool, device=sin_pos.device)
    mask[:, 0] = sin_pos >= 0.0  # RR
    mask[:, 1] = sin_pos < 0.0  # RL
    mask[sin_pos.abs() < GAIT_BIAS] = True
    return mask


def rear_stand_orientation_tracking(env, asset_cfg=None) -> torch.Tensor:
    """Follow the COMMANDED verticality (v2: the command signal IS the target --
    the policy observes it in command slot 0 and must track it up AND down),
    roll held flat -- exp kernel."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    target = env.command_manager.get_term("base_velocity").signal
    err = torch.square(target - _fwd_axis_z(asset)) + torch.square(_side_axis_z(asset))
    return torch.exp(-err / TRACKING_SIGMA)


# v6 (2026-08-16): height gets its OWN, much sharper sigma. The shared
# TRACKING_SIGMA=0.25 is calibrated for orientation-style errors (order-1 units);
# for height the error is meters SQUARED, and at 0.25 a 20cm shortfall costs only
# ~15% of the term -- which is exactly why three full runs settled into SITTING ON
# THE FOLDED CALVES ~20-30cm below target instead of standing on the feet: the
# contact sensor can't tell foot-tip contact from whole-calf contact (the foot IS
# part of the calf body in the model), so base height is the ONLY physical signal
# separating a true feet-stance from calf-sitting, and it was priced almost flat.
# 0.02 -> 0.06 (v6.1, SAME NIGHT, 6100-it postmortem): 0.02 overshot into the
# opposite failure -- vanishing gradient. It needs a full 30cm climb (quadruped
# 0.55 -> target 0.85); at 0.02 even 10cm short (most of that climb) only scores
# 0.61, 20cm short scores 0.14 -- almost no usable signal across the range the
# policy actually has to cross, only right at the very top. Three checkpoints
# (it2498/4400/6100) confirmed this empirically: height reward stayed flat at
# ~4-5% of max the entire time, identical pose in every frame, zero visible
# climbing attempt -- not "learning slowly", genuinely no gradient to climb.
# At 0.06: 10cm short ~0.85, 20cm short ~0.51, 30cm short ~0.22 -- smooth
# pressure across the whole climb, still clearly discriminates calf-sitting
# (severe shortfall) from a real stand (small shortfall).
# 0.06 -> 0.005 (v6.2, SAME NIGHT, second postmortem on the sigma itself):
# 0.06 was calibrated for the OLD 30cm climb (0.55->0.85). Lowering
# STAND_HEIGHT_TARGET to 0.62 (geometry fix, see its own comment) shrank the
# climb the ramped `target` actually spans to just 7cm (0.55->0.62) -- and
# 0.06 against a 7cm span is enormously permissive: sitting at PLAIN
# quadruped height (0cm risen, the full 7cm short) still scored exp(-0.07^2/
# 0.06)=0.92, ~92% of max, for not standing up AT ALL. Three checkpoints
# (it9200/12400/13500) confirmed empirically: height reward plateaued
# 1.2-1.9/8.0 with visually IDENTICAL pose across ~4000 iterations -- the
# policy had almost no incentive to climb the remaining 7cm since "barely
# risen" already captured nearly the whole reward. At 0.005: fully risen=1.0,
# 3cm short=0.84, 5cm short=0.61, 7cm short (not risen)=0.38 -- real pressure
# across the now-much-shorter climb. General lesson: this sigma must scale
# with (STAND_HEIGHT_TARGET - 0.55), not stay fixed across target changes.
HEIGHT_TRACKING_SIGMA = 0.005


def rear_stand_height(env, asset_cfg=None) -> torch.Tensor:
    """exp-tracking of the vertical-stand base height (flat plane -> world Z is
    ground-true). Ramped with the same clock as orientation so the two references
    never contradict each other mid-rise."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    signal = env.command_manager.get_term("base_velocity").signal
    # From the quadruped stand (0.55) up to the rear-stand target, led by the command.
    target = 0.55 + (STAND_HEIGHT_TARGET - 0.55) * signal
    err = torch.square(asset.data.root_pos_w[:, 2] - target)
    # v6: HEIGHT_TRACKING_SIGMA (own, sharp) -- see its comment above.
    return torch.exp(-err / HEIGHT_TRACKING_SIGMA)


# v6.5 (2026-08-18): owner's proposal, relayed via claude-tg-base after reading
# this file end-to-end + go2_stand's own reward set -- rear_stand_height above
# only measures root_pos_w[:,2]. Nothing prices the REAR knee angle directly, so
# "torso tilted enough to lift root height while the knee stays folded" scores
# exactly as well as a genuine feet-stance. That's precisely v6.4's final
# plateau: orientation solved (torso vertical), height stuck ~0.6-0.75/8.0,
# rear knees still visibly more bent than the Go2 reference. This constant is
# the calf-joint target that closes that hole -- see
# rear_stand_rear_leg_extension's own docstring for the full mechanism.
# -0.65 -> -1.0 (2026-08-22, owner's Go2-video reference: "нижние [rear] ноги
# ПОЛУСОГНУТЫ, упирается прямо нижними конечностями в пол" -- semi-bent, not
# near-straight). The original -0.65 was picked from the calf joint's angle
# RANGE alone ([-2.82,-0.43], "leaves ~0.22 rad margin") without checking what
# that angle actually does geometrically -- verified 2026-08-22 with a direct
# MuJoCo FK sweep (set RR_thigh/RR_calf qpos, read the foot contact geom's
# world position, not the calf BODY's -- the calf body's own origin sits AT
# the knee, angle-invariant, confirmed same "calf-independent" quirk already
# known from leg_lift's LIFT_HEIGHT_INIT FK sweep; the calf JOINT rotates a
# further foot geom 0.35m past it, which the old calf-body-based checks in
# this file never accounted for): hip-to-foot reach depends ONLY on the calf
# angle (thigh angle changes direction, not magnitude, of the reach -- an
# overall rotation about the hip). At -0.65, reach=0.674m -- 97% of the
# 0.694m fully-straight ceiling, i.e. barely bent at all despite "leaving
# margin" in raw joint-angle terms (the angle-to-reach relationship is highly
# nonlinear near the singularity, not proportional to the angle range).
# WORSE: -0.65's reach (0.674m) sits ABOVE STAND_HEIGHT_TARGET (0.62m) --
# the two anchors were fighting each other (this term pulled the leg further
# open than the height target itself needed), not reinforcing.
# -1.0 gives reach=0.626m (matches STAND_HEIGHT_TARGET directly, same FK
# sweep) at margin=0.57 rad from the -0.43 limit (26% of the full range,
# ~2.6x -0.65's margin) -- genuinely semi-bent, consistent with the height
# target instead of exceeding it, and further from the mechanical
# singularity. FK sweep script not committed (one-off scratchpad check),
# reproducible: sweep RR_thigh/RR_calf qpos in the b2_navigate MuJoCo task,
# read geom 63 (the calf body's foot-contact sphere, local pos ~[0,0,-0.35])
# world position minus RR_hip body world position.
REAR_LEG_EXTENSION_TARGET = -1.0


def rear_stand_rear_leg_extension(env, asset_cfg=None) -> torch.Tensor:
    """exp-tracking of the REAR (RL/RR) calf-joint angle toward
    REAR_LEG_EXTENSION_TARGET, ramped by the SAME command signal as
    rear_stand_height (folded at idle, extended once fully risen) but
    completely DECOUPLED from root height itself -- see
    REAR_LEG_EXTENSION_TARGET's own comment for the full diagnosis/rationale.

    Uses the file's shared TRACKING_SIGMA=0.25 rather than a new custom sigma
    -- that constant already tracks two other joint-angle terms in this file
    (rear_stand_hip_pos, rear_stand_front_legs_tuck) without needing its own
    calibration saga; height's own dedicated sigma needed THREE postmortems
    in one night (0.06->0.02->0.005) precisely because it was recalibrated in
    isolation from its actual target span -- reusing a sigma already proven
    to work on this same class of term (rad^2 joint error) is the lower-risk
    choice for a first attempt.

    Deliberately scoped to CALF only, not thigh -- the diagnosed cheat
    (pelvis tilt substituting for knee extension) is specifically a knee
    phenomenon, and thigh_joint's sign convention wasn't verified against
    URDF geometry with the same confidence as calf's (calf's range/meaning is
    directly documented via STAND_HEIGHT_TARGET's own geometry comment); a
    wrong-sign thigh term could actively fight standing rather than help.
    Revisit if calf-alone proves insufficient."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    joint_names = asset.joint_names
    ids = [i for i, n in enumerate(joint_names) if n in ("RL_calf_joint", "RR_calf_joint")]
    default = asset.data.default_joint_pos[:, ids]
    signal = env.command_manager.get_term("base_velocity").signal.unsqueeze(-1)
    target = default + (REAR_LEG_EXTENSION_TARGET - default) * signal
    err = torch.sum(torch.square(asset.data.joint_pos[:, ids] - target), dim=1)
    return torch.exp(-err / TRACKING_SIGMA)


# v7 (2026-08-18, "handstand" redesign): owner's proposal relayed via
# claude-tg-base, pointing at THIS SAME fork's own `config/others/
# unitree_a1_handstand/` task (rewards.py + rough_env_cfg.py) as a design
# reference. Their whole task is built around ONE dominant anchor --
# `handstand_feet_height_exp`, exp-tracking the AIRBORNE feet's world-Z
# height, weight 10, clearly dominant over their orientation term (-1) -- not
# root/base height at all. That's a geometrically stronger signal than
# rear_stand_height above: root height can be partly farmed by pelvis tilt
# with the rear knee still folded (v6.4's diagnosed plateau, v6.5's
# rear_leg_extension attacks the same hole from the joint-angle side); FRONT
# PAW height cannot be faked that way -- lifting the front paws to a genuine
# standing height requires the whole kinematic chain (rear knee extension +
# torso verticality) to actually be right at once, not just root_pos_w[:,2].
#
# IMPORTANT CORRECTION to the relayed suggestion: the message named
# `handstand_type="back"` as "стойка на задних лапах" (rear-leg stand, our
# goal). Verified directly against their rough_env_cfg.py + IsaacLab's own
# body-name regex resolution (isaaclab/utils/string.py uses re.fullmatch,
# confirmed empirically): handstand_type="back" sets
# `air_foot_name = "R.*_foot"`, which fullmatches ONLY "RR_foot"/"RL_foot"
# (REAR feet) -- i.e. "back" tracks the REAR feet as the ones meant to be
# airborne, meaning the robot balances on its FRONT legs (a literal
# handstand, tail end up) -- the OPPOSITE of what we want. `handstand_type
# ="front"` (air_foot_name="F.*_foot", target_gravity=[-1,0,0]) is the one
# that tracks FRONT feet as airborne, i.e. balancing on REAR/hind legs, nose
# up -- this matches our own rear_stand_orientation_tracking's own documented
# convention exactly ("nose-up vertical" -> rear feet as the support base).
# Ported the FRONT-airborne logic below, not "back" as literally suggested --
# flagged explicitly back to the owner/claude-tg-base for review since this
# was the load-bearing parameter of the whole port.
#
# Target height: not derived with STAND_HEIGHT_TARGET's clean 2-link-IK
# confidence (that needs the WHOLE kinematic chain incl. body length, not
# just one leg) -- reasoned estimate only. b2.urdf: base_link -> front hip
# offset is 0.3285m along body-local X; once vertical that offset becomes
# vertical, so front-hip height once fully risen ~= STAND_HEIGHT_TARGET
# (0.62) + 0.3285 =~ 0.95m. Target set a bit under that (0.85m) to leave the
# front leg some functional bend rather than assume full extension. FIRST
# GUESS -- expect this needs the same postmortem-driven recalibration
# STAND_HEIGHT_TARGET/HEIGHT_TRACKING_SIGMA both needed tonight; 0.15 is the
# idle (4-leg stance) front-paw height baseline the ramp starts from.
FRONT_FEET_HEIGHT_TARGET = 0.85
FRONT_FEET_IDLE_HEIGHT = 0.15
# Sigma: reused from their own reference value (std=sqrt(0.25) -> sigma=0.25
# in this file's exp(-err/sigma) convention) rather than derived from
# scratch -- it's the literal value their working reference uses for this
# exact term, applied to a similarly-scaled (sub-meter) height-error span;
# also happens to equal this file's own shared TRACKING_SIGMA, so no new
# untested constant is introduced. Lower-risk than a from-scratch derivation
# after tonight's THREE-postmortem sigma saga on rear_stand_height itself.
FRONT_FEET_HEIGHT_SIGMA = TRACKING_SIGMA


def rear_stand_front_feet_height(env, asset_cfg=None) -> torch.Tensor:
    """exp-tracking of the FRONT (FL/FR) paw world-Z height toward
    FRONT_FEET_HEIGHT_TARGET, ramped by the same command signal as height/
    orientation (idle baseline -> full target once risen). See
    FRONT_FEET_HEIGHT_TARGET's own comment for the full "handstand" redesign
    rationale and the handstand_type correction. NEW dominant anchor --
    rear_stand_height stays active as a secondary signal (weight reduced),
    not replaced outright."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    body_names = asset.body_names
    ids = [body_names.index("FL_calf"), body_names.index("FR_calf")]
    feet_height = asset.data.body_pos_w[:, ids, 2]
    signal = env.command_manager.get_term("base_velocity").signal.unsqueeze(-1)
    target = FRONT_FEET_IDLE_HEIGHT + (FRONT_FEET_HEIGHT_TARGET - FRONT_FEET_IDLE_HEIGHT) * signal
    err = torch.sum(torch.square(feet_height - target), dim=1)
    return torch.exp(-err / FRONT_FEET_HEIGHT_SIGMA)


def rear_stand_front_feet_on_air(env, sensor_cfg=None) -> torch.Tensor:
    """Ported from unitree_a1_handstand's own handstand_feet_on_air (front
    feet in first-air state). Gated on risen (signal>0.9, same threshold as
    rear_stand_front_legs_tuck/rear_stand_com_over_support) -- their original
    is ungated because their task has no idle/four-leg phase at all; ours
    does, and front feet belong ON the ground during idle/rise."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.all(first_air, dim=1).float()
    risen = (env.command_manager.get_term("base_velocity").signal > 0.9).float()
    return reward * risen


def rear_stand_front_feet_air_time(env, sensor_cfg=None, threshold: float = 3.0) -> torch.Tensor:
    """Ported from unitree_a1_handstand's own handstand_feet_air_time
    (rewards sustained front-feet air time past `threshold`). Their own
    threshold=5.0 was calibrated against a fixed 10s episode with no idle
    phase; ours holds 8-14s (hold_time_range) on top of a 2s rise, so 3.0 is
    a scaled-down FIRST GUESS, not a derived value -- same gating as
    rear_stand_front_feet_on_air, for the same reason."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    risen = (env.command_manager.get_term("base_velocity").signal > 0.9).float()
    return reward * risen


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


def rear_stand_idle_joint_pose(env, asset_cfg=None) -> torch.Tensor:
    """L2 penalty on joint deviation from the default quadruped pose, active ONLY
    during idle (command signal ~0). Added 2026-08-09 (user bench-test on
    model_50598: right hind leg visibly twisted forward -- both thigh and calf --
    in the resting four-leg pose; 30599 and 4100 look correct in the same pose).

    Root cause: stand_still_without_cmd and joint_pos_penalty -- the two terms
    that normally hold the default pose everywhere -- are BOTH zeroed for this
    entire task (see __post_init__ below, "pulls to 4-leg default pose" --
    conflicts with rearing). That leaves idle joint symmetry completely
    unpriced: nothing has ever constrained individual joint angles while on
    four legs, for the whole life of this task -- an unpriced free variable
    that could drift at any point in any run; this one just happened to
    visibly manifest it. Gated the same way as rear_stand_idle_still (signal
    < 0.1) so it only fires on four legs, never fighting rise/hold/walk/descend."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    err = torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=1)
    idle = (env.command_manager.get_term("base_velocity").signal < 0.1).float()
    return err * idle


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
    """REAR feet are the support. Standing/rising/descending: both down (mean),
    unchanged since v2. v4 change (was v3's at-least-one-down max, bench-confirmed
    on 30599/50598 as "почти нулевое перемещение" -- barely steps): walking now
    pays for matching the gait-clock's alternating stance/swing schedule exactly
    (go2_stand's own _reward_rear_feet_contact_and_air, ported) instead of any
    contact pattern satisfying "at least one down" -- the max-based v3 version let
    the policy shuffle both feet down with no real stepping rhythm, since two feet
    planted always satisfied it for free."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0
    walking = (torch.abs(env.command_manager.get_term("base_velocity").command[:, 1]) > 0.05).float()
    mask = _gait_mask(env)
    gait_reward = torch.sum((in_contact & mask).float() + (~in_contact & ~mask).float(), dim=1)
    return walking * gait_reward + (1.0 - walking) * in_contact.float().mean(dim=1)


def rear_stand_feet_clearance(env, sensor_cfg=None) -> torch.Tensor:
    """v4: reward the SWING-phase rear foot (calf-body world Z as the foot-height
    proxy -- same simplification already used by rear_stand_com_over_support/
    rear_stand_stance_width in this file) for reaching a target height --
    go2_stand's own _reward_feet_clearance, ported. Without this, "stepping" can
    satisfy the gait-contact reward with a foot barely off the ground (any
    non-contact counts as swing); this prices the swing motion itself. Target is
    an uncalibrated first guess (same convention as STAND_HEIGHT_TARGET/
    CROUCH_TARGET_HEIGHT elsewhere in this file) -- it shapes gradient direction,
    not a hard constraint, so a rough value is fine to start."""
    asset = env.scene["robot"]
    body_names = asset.body_names
    rr, rl = body_names.index("RR_calf"), body_names.index("RL_calf")
    foot_z = torch.stack([asset.data.body_pos_w[:, rr, 2], asset.data.body_pos_w[:, rl, 2]], dim=1)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0
    swing = ~in_contact & ~_gait_mask(env)
    err = torch.abs(foot_z - FEET_CLEARANCE_TARGET)
    walking = (torch.abs(env.command_manager.get_term("base_velocity").command[:, 1]) > 0.05).float()
    return torch.sum(torch.exp(-err) * swing.float(), dim=1) * _risen_mask(env, asset) * walking


def rear_stand_foot_slip(env, sensor_cfg=None) -> torch.Tensor:
    """v4: penalize rear-foot horizontal velocity while in contact -- go2_stand's
    own _reward_foot_slip, ported. Ungated (fires in any phase, like the
    original) -- a planted foot sliding is bad whether idle, rising, or walking."""
    asset = env.scene["robot"]
    body_names = asset.body_names
    rr, rl = body_names.index("RR_calf"), body_names.index("RL_calf")
    foot_speed = torch.stack(
        [asset.data.body_lin_vel_w[:, rr, 0:2].norm(dim=1), asset.data.body_lin_vel_w[:, rl, 0:2].norm(dim=1)], dim=1
    )
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0
    return torch.sum(foot_speed * in_contact.float(), dim=1)


def rear_stand_hip_pos(env, asset_cfg=None) -> torch.Tensor:
    """v4: exp-tracking of default HIP joint angles, unconditional every phase --
    go2_stand's own _reward_hip_pos, ported. This gait's propulsion is entirely
    thigh/calf; locking the hips removes 4 of 12 DOF from what the policy needs to
    coordinate, and is a second, always-on backstop against the same idle
    free-variable drift rear_stand_idle_joint_pose targets in idle only (bench:
    twisted RR leg on model_50598)."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    joint_names = asset.joint_names
    hip_ids = [i for i, n in enumerate(joint_names) if n.endswith("_hip_joint")]
    err = torch.sum(torch.square(asset.data.joint_pos[:, hip_ids] - asset.data.default_joint_pos[:, hip_ids]), dim=1)
    return torch.exp(-err / TRACKING_SIGMA)


def rear_stand_front_legs_tuck(env, asset_cfg=None) -> torch.Tensor:
    """exp-tracking of the FRONT legs' thigh+calf joints to their default
    angles while risen (added 2026-08-12, bench on the v5 13600 checkpoint:
    right front paw raised and waving, both front legs in visible tremor).

    Pricing hole: once vertical, the front legs are airborne by design --
    front_feet_contact prices touching the ground, hip_pos anchors the 4 hip
    ab/ad joints, but the front THIGH and CALF joints while risen were never
    priced by anything (stand_still_without_cmd/joint_pos_penalty are zeroed
    task-wide, idle_joint_pose gates to signal<0.1) -- a free variable for the
    entire life of the task, waving/tremor is just where it drifted. (The
    bench clock mismatch -- 13600 was tested against a 2x-fast gait metronome,
    see b2_policy.py's own note -- may have amplified the tremor, but the hole
    is real regardless and costs nothing to close.) Same exp/TRACKING_SIGMA
    idiom as rear_stand_hip_pos, gated on the commanded signal so the rise and
    descend transitions stay free to move the legs."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    joint_names = asset.joint_names
    ids = [
        i for i, n in enumerate(joint_names)
        if n in ("FL_thigh_joint", "FL_calf_joint", "FR_thigh_joint", "FR_calf_joint")
    ]  # fmt: skip
    err = torch.sum(torch.square(asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]), dim=1)
    risen = (env.command_manager.get_term("base_velocity").signal > 0.9).float()
    return torch.exp(-err / TRACKING_SIGMA) * risen


def rear_stand_no_flight(env, sensor_cfg=None) -> torch.Tensor:
    """Penalty when BOTH rear feet are out of contact while risen (added
    2026-08-12, bench on 13600: commanded forward, the robot doesn't step --
    it micro-HOPS in place). A biped WALK always keeps at least one foot
    planted; a hop launches both. Nothing priced that: the gait-schedule
    reward only scores per-foot contact matching, and a well-timed hop can
    roughly satisfy an alternating schedule around the double-support windows
    while going nowhere. Walking (the desired gait) pays this zero by
    construction; only genuinely ballistic ticks pay."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0
    both_off = (~in_contact).all(dim=1).float()
    risen = (env.command_manager.get_term("base_velocity").signal > 0.9).float()
    return both_off * risen


def rear_stand_low_speed(env, asset_cfg=None) -> torch.Tensor:
    """v4: coarse threshold penalty on walking speed vs command -- go2_stand's own
    _reward_low_speed, ported. rear_stand_walk_tracking's exp-kernel is forgiving
    near zero (a stationary robot still collects ~70% of max reward against a
    0.3 m/s command at this file's own TRACKING_SIGMA=0.25) -- bench-confirmed
    "barely steps" on both 30599 and 50598 despite walk_tracking being this
    task's highest-weighted v3 term. This adds a harder floor: below half the
    commanded speed pays -1, the wrong direction pays -2, hitting the commanded
    band (0.5x-1.2x) pays +1.2."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    cmd_vx = env.command_manager.get_term("base_velocity").command[:, 1]
    v_walk = torch.sum(asset.data.root_lin_vel_w[:, 0:2] * _walk_dir_xy(asset), dim=1)
    active = torch.abs(cmd_vx) > 0.05
    too_slow = v_walk.abs() < 0.5 * cmd_vx.abs()
    too_fast = v_walk.abs() > 1.2 * cmd_vx.abs()
    wrong_way = torch.sign(v_walk) != torch.sign(cmd_vx)
    reward = torch.where(too_slow, -1.0, 0.0)
    reward = torch.where(~too_slow & ~too_fast, 1.2, reward)
    reward = torch.where(wrong_way, -2.0, reward)
    return reward * active.float() * _risen_mask(env, asset)


def rear_stand_turn_tracking(env, asset_cfg=None) -> torch.Tensor:
    """v4: exp-tracking of the commanded yaw rate (slot 4) while risen --
    go2_stand's own tracking_ang_vel, ported (turning IS a bench-confirmed trained
    capability there, per its own commit history "адекватная ходьба и управление
    влево или вправо"). v3 never trained this at all -- the bench driver
    repurposed the yaw stick as DESCEND purely because nothing else used it."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    cmd_wz = env.command_manager.get_term("base_velocity").command[:, 4]
    err = torch.square(cmd_wz - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-err / TRACKING_SIGMA) * _risen_mask(env, asset)


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
    is gated to signal<0.1).
    v4: yaw is no longer unconditionally priced here -- while a turn is actually
    commanded, rear_stand_turn_tracking owns yaw (pricing it here too would
    directly fight that reward); yaw drift during walk/stand cycles is still
    priced exactly as before."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    walk_dir = _walk_dir_xy(asset)
    v_xy = asset.data.root_lin_vel_w[:, 0:2]
    v_perp = v_xy[:, 0] * walk_dir[:, 1] - v_xy[:, 1] * walk_dir[:, 0]
    cmd_wz = env.command_manager.get_term("base_velocity").command[:, 4]
    not_turning = (torch.abs(cmd_wz) < 0.05).float()
    w_z = torch.square(asset.data.root_ang_vel_w[:, 2]) * not_turning
    return (torch.square(v_perp) + w_z) * _risen_mask(env, asset)


def rear_stand_action_rate_l2_clamped(env) -> torch.Tensor:
    """Same as isaaclab.envs.mdp.action_rate_l2 (L2-squared rate of change of the
    action), but the per-dimension diff is clamped BEFORE squaring -- rear_stand-
    only override of the shared base term (added 2026-08-10 night, user-approved
    after two live incidents: it18694-19894 on 2026-08-10_03-35-39 hit
    action_rate_l2 raw sums in the thousands, then a SECOND, larger episode on
    2026-08-10_08-32-02 (after an entropy_coef fix that only partially helped)
    reached raw sums whose SQUARE, at weight=-0.01, printed as -1M+ Episode_Reward
    and blew value_function loss past 700M -- same root cause diagnosed for
    jump_motor_speed_violation back on 2026-08-08: a rare per-env physics-glitch
    tick can spike ONE action dimension's frame-to-frame delta arbitrarily far,
    and an unbounded squared penalty has no ceiling on how hard that one glitch
    can wreck a whole batch's value-function target.

    Same clamp-before-squaring discipline as jump_motor_speed_violation/
    jump_airborne_leg_stillness's own postmortem fixes. Bound of 3.0 per
    dimension is generous relative to any healthy observed diff (normal
    Episode_Reward/action_rate_l2 stayed in the single digits to tens all
    night -- raw per-dimension diffs well under 1) while capping the worst
    possible single-tick contribution at 9 per dimension instead of unbounded.

    Deliberately NOT edited in isaaclab's own mdp.action_rate_l2 (shared library
    function, used verbatim by rough/crawl/jump/vision too) -- overriding only
    here keeps those other tasks' already-tuned dynamics untouched, and confines
    the blast radius of an unreviewed clamp bound to the one task that actually
    needed it."""
    diff = (env.action_manager.action - env.action_manager.prev_action).clamp(-3.0, 3.0)
    return torch.sum(torch.square(diff), dim=1)


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

        # v2: the stance-cycle command replaces the velocity command wholesale.
        # v4: command grew 3->5 slots (stand signal, walk vx, sin/cos gait phase,
        # turn wz) -- flows straight into the existing velocity_commands obs term
        # (mdp.generated_commands reads the whole vector) with no ObservationsCfg
        # change, but this DOES change the policy's input width, so a v4
        # checkpoint cannot warm-start from any v2/v3 one.
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

        # 2026-08-10 night: rear_stand-only override of the shared action_rate_l2
        # base term -- see rear_stand_action_rate_l2_clamped's own docstring for
        # the two-incident postmortem (up to -1M+ Episode_Reward, value_function
        # loss past 700M, from one rare per-env glitch tick, unbounded). Same
        # weight as rough's own default (-0.01) -- only the formula changed
        # (clamp before squaring), not the pricing.
        self.rewards.action_rate_l2 = RewTerm(func=rear_stand_action_rate_l2_clamped, weight=-0.01)

        # -- the rear-stand objective (weights mirror go2_stand's own proportions)
        # 5 -> 8 (2026-08-06, both v2 starts parked half-risen): with no falls at
        # all (episodes flat 1000), the moat around the ignore-the-command optimum
        # is not risk -- it's gradient weakness; make following the signal pay
        # decisively more than parking.
        self.rewards.rear_stand_orientation_tracking = RewTerm(
            func=rear_stand_orientation_tracking, weight=8.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        self.rewards.rear_stand_height = RewTerm(
            # 3.0 -> 8.0 (v6 stand-only): height is the ONE signal separating a
            # true feet-stance from sitting on folded calves (see
            # HEIGHT_TRACKING_SIGMA's comment) -- promote it to the same dominant
            # tier as orientation (8.0) so "stand tall on the feet" outbids the
            # safe low crouch, paired with the sharp per-term sigma.
            # 8.0 -> 4.0 (v7, 2026-08-18, handstand redesign): root height is
            # no longer the SOLE/dominant standing signal -- demoted to a
            # secondary anchor now that rear_stand_front_feet_height (10.0,
            # below) owns primary authority. Not zeroed: still a useful,
            # cheap-to-satisfy-honestly signal, just no longer load-bearing
            # alone.
            func=rear_stand_height, weight=4.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        # v7 (2026-08-18, handstand redesign): the NEW dominant anchor -- see
        # FRONT_FEET_HEIGHT_TARGET's own comment for the full mechanism/
        # rationale/handstand_type correction. Weight 10.0 mirrors
        # unitree_a1_handstand's own proportions (their feet_height_exp=10 vs
        # orientation=-1, clearly dominant); scaled here against our own
        # orientation=8/height=4 rather than copied blindly.
        self.rewards.rear_stand_front_feet_height = RewTerm(
            func=rear_stand_front_feet_height, weight=10.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        self.rewards.rear_stand_front_feet_on_air = RewTerm(
            func=rear_stand_front_feet_on_air,
            weight=5.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"])},
        )
        self.rewards.rear_stand_front_feet_air_time = RewTerm(
            func=rear_stand_front_feet_air_time,
            weight=5.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["FL_calf", "FR_calf"]),
                "threshold": 3.0,
            },
        )
        self.rewards.rear_stand_com_over_support = RewTerm(
            func=rear_stand_com_over_support, weight=2.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        # v6.5 (2026-08-18): closes the "pelvis tilt substitutes for knee
        # extension" hole diagnosed in v6.4's postmortem -- see
        # REAR_LEG_EXTENSION_TARGET's own comment for the full mechanism.
        # Weighted just under height/orientation (8.0) -- this is a primary
        # co-objective (height alone is gameable without it), not a minor
        # backstop like hip_pos/front_legs_tuck (3.0/2.0).
        self.rewards.rear_stand_rear_leg_extension = RewTerm(
            func=rear_stand_rear_leg_extension, weight=6.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        # v4 (2026-08-09): unconditional hip anchor, go2_stand's own hip_pos
        # ported -- see the function's own docstring. Weight matched to
        # rear_stand_height's own 3.0 (same "secondary anchor" scale as
        # orientation's 8/height's 3 proportions, per this block's own comment).
        self.rewards.rear_stand_hip_pos = RewTerm(
            func=rear_stand_hip_pos, weight=3.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        # New 2026-08-12 (bench on v5's 13600: front-leg tremor + a raised
        # waving paw -- see rear_stand_front_legs_tuck's own docstring).
        # Weight 2.0 -- secondary-anchor scale, same tier as feet_clearance.
        # 2.0 -> 0 (v7, 2026-08-18, handstand redesign): DIRECTLY CONFLICTS
        # with the new dominant rear_stand_front_feet_height -- this term
        # anchors front thigh+calf toward the DEFAULT (folded, low) quadruped
        # angle while risen, exactly opposing "lift the front paws high".
        # front_feet_height now gives the front legs a genuine job (reach a
        # target height) instead of leaving them a free variable, which was
        # this term's whole original purpose -- superseded, not just
        # redundant.
        self.rewards.rear_stand_front_legs_tuck = RewTerm(
            func=rear_stand_front_legs_tuck, weight=0.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        # New 2026-08-12 (bench on v5's 13600: micro-hopping in place instead
        # of stepping -- see rear_stand_no_flight's own docstring).
        # -2.0 -> -6.0 (v5.2, 2026-08-16, owner bench verdict on the v5.1 FINAL
        # 19999: "просто встаёт и пытается прыгать мелко на двух ногах, никакой
        # походки". Sequence-frame review confirmed hopping, not walking (see
        # TRAINING_STATE.md ~12:00/TRAIN_RESEARCH's "stop-frame vs walking"
        # entry). Root cause: `rear_stand_no_flight` held FLAT at -1.0..-1.1
        # episode-mean the ENTIRE 20000-it run -- the fix term never gained
        # traction. Economics check against rear_stand_rear_feet_contact
        # (weight=3.0, below): a hop landing during a SINGLE-support window
        # still matches the gait mask on the one foot that's ALSO supposed to
        # be airborne (_gait_mask is per-foot, not "both or neither"), earning
        # partial credit (weight*1=3.0) that -2.0 here didn't outweigh --
        # hopping was cheaper than learning genuine single-leg balance.
        # -6.0 flips that: hop nets 3.0-6.0=-3.0 on this pair alone, a clean
        # single-support step still nets the full 3.0*2=6.0 walking bonus
        # untouched (no_flight only fires when BOTH feet are off).
        self.rewards.rear_stand_no_flight = RewTerm(
            func=rear_stand_no_flight,
            weight=-6.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"])},
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
        # New 2026-08-09 (user bench-test on model_50598: twisted RR leg at idle --
        # see the function's own docstring for the root-cause analysis). Weight
        # matched to rear_stand_idle_still's own -2.0 -- same "idle discipline"
        # scale, no reason to weight joint symmetry differently than idle drift.
        self.rewards.rear_stand_idle_joint_pose = RewTerm(
            func=rear_stand_idle_joint_pose, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        self.rewards.rear_stand_rear_feet_contact = RewTerm(
            func=rear_stand_rear_feet_contact,
            # 1.0 -> 3.0 (v5): the alternating contact schedule is go2_stand's own
            # DOMINANT walking lever (their w=4); at 1.0 here it was an afterthought
            # the stand-still baseline could shrug off (see v5 module comment,
            # cause 4). Still below orientation's 8 -- falling must never win.
            weight=3.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"])},
        )

        # -- v3: bipedal walking while vertical (command slot 1, hold phase).
        # 3.0 -> 5.0 (2026-08-07, run 08-24-06 plateaued ~5000 iters, noise_std
        # DECLINING not climbing -- unlike jump's own plateaus this isn't a
        # temporary lull, it's converged economics: orientation_tracking already
        # sits near its ceiling (7.6/8) while walk_tracking stalled at ~1.2/~3 max
        # -- committing to a real stride risks the orientation term for too little
        # walking payoff, so the policy plays it safe and barely steps. Still below
        # orientation's 8 -- falling over to chase velocity must never win.
        self.rewards.rear_stand_walk_tracking = RewTerm(
            func=rear_stand_walk_tracking, weight=5.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        self.rewards.rear_stand_walk_drift = RewTerm(
            func=rear_stand_walk_drift, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        # -- v4: gait-quality shaping for the walk (go2_stand's own feet_clearance/
        # foot_slip/low_speed, ported -- see each function's own docstring).
        self.rewards.rear_stand_feet_clearance = RewTerm(
            func=rear_stand_feet_clearance,
            # 1.0 -> 2.0 (v5): scaled up alongside rear_feet_contact's own 3.0 --
            # the swing half of the same gait mechanism (see v5 module comment).
            weight=2.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"])},
        )
        self.rewards.rear_stand_foot_slip = RewTerm(
            func=rear_stand_foot_slip,
            weight=-2.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["RL_calf", "RR_calf"])},
        )
        # go2_stand's own low_speed carried weight=0.005 against ~1-2 magnitude raw
        # reward (a light nudge on top of the gait-clock mechanism, which is the
        # actual dominant lever there). Sized proportionally into this file's own
        # weight scale (walk_tracking=5.0) instead of copying their absolute number.
        # 0.5 -> 1.5 (v5): at 0.5 the "-1 for standing during a walk command" floor
        # cost -0.5/s against orientation's safe +8/s -- an ignorable tax (see v5
        # module comment, cause 4). At 1.5 the too-slow floor is -1.5/s and the
        # in-band bonus +1.8/s: a 3.3/s swing for actually striding.
        self.rewards.rear_stand_low_speed = RewTerm(
            func=rear_stand_low_speed, weight=1.5, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        # -- v4: turning (command slot 4, hold phase) -- go2_stand's own
        # tracking_ang_vel, ported. Weight matched to walk_tracking's own 5.0 --
        # no reason to favor one bipedal skill's gradient over the other.
        self.rewards.rear_stand_turn_tracking = RewTerm(
            func=rear_stand_turn_tracking, weight=5.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeB2RearStandRoughEnvCfg":
            self.disable_zero_weight_rewards()
