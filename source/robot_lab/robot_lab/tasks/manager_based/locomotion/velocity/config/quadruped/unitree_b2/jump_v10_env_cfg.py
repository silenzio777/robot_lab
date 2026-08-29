# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""JUMP v10 -- from-scratch minimal redesign (2026-08-26, owner's direct order).

Owner's own words, verbatim reasoning for this restart: the old jump_env_cfg.py
(30+ reward terms, a long chain of exploit-fixes -- rearing, diving, crouch-
turnaround gaming, landing-package free-riding, yaw-drift-as-distance-exploit)
never converged to a clean jump. Rather than patch that file again, the owner
described the motion directly from a real reference (frame-by-frame, his own
observation of the vendor B2 jump video) and asked for ONLY the first two
phases to be trained, nothing else, so the biomechanics get right before any
flight/landing/distance shaping is added back:

    0. StandFix   -- standing still on all fours (warm-start already knows this).
    1. Crouch     -- fold down DEEP and LEVEL (not lopsided, not sitting back
                      onto the hindquarters -- "присела полностью").
    2. Launch     -- strong, fast, SYMMETRIC extension of both leg pairs,
                      producing real forward+upward velocity ("на пол-корпуса
                      вперёд" + "30-40см клиренса" at the very start of flight).

Explicitly OUT OF SCOPE this pass (owner: "Пока только это!"): flight-phase leg
pose, landing absorption, distance-to-target, height ladder. What happens after
liftoff is not gated or rewarded here at all -- the cycle simply returns to idle
once the robot has genuinely left the ground (or a hard cap expires), and physics
takes it from there. jump_env_cfg.py (the old 30-term file) is left completely
untouched as a reference/fallback -- this is a NEW gym task, not an edit.

Design collaboration: base (claude-tg-base) proposed the reward skeleton and the
crouch/launch reward split; the owner corrected the launch objective mid-design
from vertical-only to forward+vertical (matching reference frame 2, "на
пол-корпуса вперёд" at the same clearance the vertical component targets).

WARM-START: `stage_a_standing` (LEG_LIFT v9.6, model_7100) -- the owner's own
bench-CONFIRMED standing checkpoint, not any prior jump-line donor (every prior
jump donor already carries drifted/asymmetric standing habits from its own long
exploit-fix history -- warm-starting from those would re-import exactly the
asymmetry this restart exists to avoid). Same 45-obs/12-action/3-command-slot
layout as every other skill in this family -- warm-start is dimension-
compatible regardless of the donor's own original command semantics (same
established convention this whole codebase already relies on, see the original
jump_env_cfg.py's own module docstring).

Ballistics check (owner's target: 30-40cm foot clearance at the first flight
frame, reference frame 2): h = v_z^2 / (2*g) =>
    v_z(0.30m) = sqrt(2*9.81*0.30) = 2.426 m/s
    v_z(0.40m) = sqrt(2*9.81*0.40) = 2.802 m/s
so a peak (honest, min-over-4-hip) vertical launch velocity of ~2.4-2.8 m/s is
the physical target the launch reward should pull toward -- this is where the
JUMP_V10_LAUNCH_VZ_CLAMP below comes from, not a guess.
"""

import torch

import isaaclab.utils.math as math_utils
import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .jump_env_cfg import FOOT_GEOM_LOCAL_OFFSET
from .rough_env_cfg import UnitreeB2RoughEnvCfg


class JumpCrouchLaunchCommand(CommandTerm):
    """Three-state per-env cycle: idle -> crouch -> launch -> idle ...

    Deliberately NOT the old JumpPulseCommand's fixed-duration window: this is a
    from-scratch minimal design (owner: "Пока только это!"), so the state
    machine only has what THIS pass needs -- no landing slice, no had_flight
    exploit-latch, no crouch-hold depth-conditional extension, no distance
    bookkeeping. Crouch is a fixed duration (deep fold needs time to load, not
    an event); launch ends on a genuine liftoff event (not a timer) so the
    reward window closes right when the thing it was rewarding actually
    happened, with a hard cap as a pure safety net for early training when no
    real liftoff occurs at all.
    """

    cfg: "JumpCrouchLaunchCommandCfg"

    def __init__(self, cfg: "JumpCrouchLaunchCommandCfg", env) -> None:
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        n = self.num_envs
        self._command = torch.zeros(n, 3, device=self.device)
        # 0=idle, 1=crouch, 2=launch.
        self.state = torch.zeros(n, dtype=torch.long, device=self.device)
        self.state_elapsed = torch.zeros(n, device=self.device)
        self.idle_duration = torch.zeros(n, device=self.device)
        self.phase = torch.zeros(n, device=self.device)
        self.crouch_active = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.launch_active = torch.zeros(n, dtype=torch.bool, device=self.device)

        # Launch-direction anchor, captured ONCE the instant launch begins --
        # same idiom as the old jump_env_cfg.py's v7d postmortem fix
        # (launch_yaw): a policy free to recompute "forward" from its OWN
        # live yaw every step can turn a yaw-spin into free forward-velocity
        # credit. Owner's mid-design correction added a forward-velocity
        # launch term specifically, so this exploit is back in scope from day
        # one -- fixing the direction at launch-start closes it structurally
        # instead of hoping training avoids it (the old line only added this
        # AFTER measuring the exploit live -- doing it up front here).
        self.launch_yaw = torch.zeros(n, device=self.device)
        self._launch_yaw_captured = torch.zeros(n, dtype=torch.bool, device=self.device)

        # Crouch-entry pose anchor, captured ONCE the instant crouch begins --
        # same latch idiom as launch_yaw directly above (base's design,
        # 2026-08-29). jump_v10_crouch_pose's phase=0 end of its ramp used to
        # be a FIXED assumed-standing constant (STAND_THIGH_TARGET/STAND_
        # CALF_TARGET) -- found to drift out of sync when idle's own achieved
        # pose shifted over training (idle calf drifted -1.5->-1.0,
        # base_z 0.54->0.635m, one full night in), creating an artificial
        # ramp discontinuity right at phase=0 instead of the intended
        # zero-pressure start. Capturing the REAL joint angles at the actual
        # idle->crouch transition makes phase=0 always pressure-free by
        # construction, regardless of whatever idle equilibrium the policy
        # currently holds -- decouples crouch-shaping from idle-drift
        # permanently instead of re-tuning constants after every future
        # shift.
        self.crouch_entry_thigh = torch.zeros(n, 4, device=self.device)
        self.crouch_entry_calf = torch.zeros(n, 4, device=self.device)
        self._crouch_entry_captured = torch.zeros(n, dtype=torch.bool, device=self.device)
        joint_names = self.robot.joint_names
        self._thigh_joint_ids = torch.tensor(
            [joint_names.index(f"{leg}_thigh_joint") for leg in ("FL", "FR", "RL", "RR")],
            device=self.device,
        )
        self._calf_joint_ids = torch.tensor(
            [joint_names.index(f"{leg}_calf_joint") for leg in ("FL", "FR", "RL", "RR")],
            device=self.device,
        )

        # Contact + foot-clearance plumbing, same resolution idiom as
        # jump_env_cfg.py's JumpPulseCommand (preserve_order=True matters --
        # SceneEntityCfg.resolve() does NOT rewrite body_names to the matched
        # names, only body_ids, in regex-match order -- see that file's own
        # __init__ comment for the full explanation of why find_bodies with
        # preserve_order is used directly instead).
        self._feet_sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_calf")
        self._feet_sensor_cfg.resolve(self._env.scene)
        contact_sensor_entity = self._env.scene[self._feet_sensor_cfg.name]
        feet_sensor_order, _ = contact_sensor_entity.find_bodies(
            ("FL_calf", "FR_calf", "RL_calf", "RR_calf"), preserve_order=True
        )
        self._feet_sensor_order = torch.tensor(feet_sensor_order, device=self.device)
        body_names = self.robot.body_names
        self._calf_body_ids = torch.tensor(
            [body_names.index(n) for n in ("FL_calf", "FR_calf", "RL_calf", "RR_calf")],
            device=self.device,
        )
        self._hip_body_ids = torch.tensor(
            [body_names.index(n) for n in ("FL_hip", "FR_hip", "RL_hip", "RR_hip")],
            device=self.device,
        )
        self._foot_local_offset = torch.tensor(FOOT_GEOM_LOCAL_OFFSET, device=self.device)
        self.min_foot_clearance = torch.zeros(n, device=self.device)
        self.all_airborne = torch.zeros(n, dtype=torch.bool, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        # Fraction of envs in each state -- cheap TensorBoard liveness signal,
        # nothing decision-driving (same role as the old file's
        # window_active_ratio).
        self.metrics["crouch_active_ratio"] = self.crouch_active.float()
        self.metrics["launch_active_ratio"] = self.launch_active.float()

    def _resample_command(self, env_ids):
        # This state machine owns its own cycling (see _update_command) --
        # not the base class's time_left countdown. Push time_left far past
        # any real episode length so CommandTerm's own auto-resample
        # (time_left <= 0) never fires and never fights this class's own
        # idle->crouch->launch->idle transitions. Still called correctly at
        # every episode reset (env_ids), which is exactly when a fresh idle
        # state should begin.
        self.state[env_ids] = 0
        self.state_elapsed[env_ids] = 0.0
        self.idle_duration[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self.cfg.idle_time_range
        )
        self.time_left[env_ids] = 1.0e9
        self._launch_yaw_captured[env_ids] = False
        self._crouch_entry_captured[env_ids] = False

    def _update_command(self):
        step_dt = self._env.step_dt
        self.state_elapsed = self.state_elapsed + step_dt

        # -- ground-truth contact/clearance (needed for the launch-exit event) --
        contact_sensor = self._env.scene.sensors[self._feet_sensor_cfg.name]
        in_contact = contact_sensor.data.current_contact_time[:, self._feet_sensor_order] > 0.0
        all_airborne = ~in_contact.any(dim=1)
        # Stored persistently (2026-08-29, base's design, step (а) of the
        # somersault fix -- see jump_v10_level/jump_v10_yaw_rate's own updated
        # gate below): this was a local var before, recomputed but discarded
        # every step. bad_orientation (step (б), same day) already stops a
        # tumble from completing, but level/yaw_rate's own gate still only
        # covers crouch_active|launch_active -- exposing this as ground truth
        # for "still physically airborne right now" (not just "launch's own
        # timer hasn't expired yet") lets those two terms cover the real
        # post-launch inertial-flight window too, closing the gap at its
        # source instead of only backstopping the consequence.
        self.all_airborne = all_airborne
        calf_pos_w = self.robot.data.body_pos_w[:, self._calf_body_ids, :]
        calf_quat_w = self.robot.data.body_quat_w[:, self._calf_body_ids, :]
        n_envs = calf_pos_w.shape[0]
        local_offset = self._foot_local_offset.expand(n_envs, 4, 3)
        world_offset = math_utils.quat_apply(
            calf_quat_w.reshape(-1, 4), local_offset.reshape(-1, 3)
        ).reshape(n_envs, 4, 3)
        foot_pos_w = calf_pos_w + world_offset
        self.min_foot_clearance = foot_pos_w[:, :, 2].min(dim=1).values

        idle_mask = self.state == 0
        crouch_mask = self.state == 1
        launch_mask = self.state == 2

        # idle -> crouch
        to_crouch = idle_mask & (self.state_elapsed >= self.idle_duration)
        self.state = torch.where(to_crouch, torch.ones_like(self.state), self.state)
        self.state_elapsed = torch.where(to_crouch, torch.zeros_like(self.state_elapsed), self.state_elapsed)

        # crouch -> launch
        to_launch = crouch_mask & (self.state_elapsed >= self.cfg.crouch_duration)
        self.state = torch.where(to_launch, torch.full_like(self.state, 2), self.state)
        self.state_elapsed = torch.where(to_launch, torch.zeros_like(self.state_elapsed), self.state_elapsed)

        # launch -> idle: real liftoff (all 4 feet genuinely clear of the
        # ground, contact sensor + clearance both agree) held for a short
        # buffer, OR a hard-cap timeout if no real liftoff ever happens this
        # cycle (safety net for early training -- a policy that never pushes
        # must still cycle back to idle and try again, not get stuck).
        liftoff_now = launch_mask & all_airborne & (self.min_foot_clearance > self.cfg.liftoff_clearance_eps)
        to_idle = launch_mask & (
            (liftoff_now & (self.state_elapsed >= self.cfg.launch_min_before_idle))
            | (self.state_elapsed >= self.cfg.launch_hard_cap)
        )
        self.state = torch.where(to_idle, torch.zeros_like(self.state), self.state)
        self.state_elapsed = torch.where(to_idle, torch.zeros_like(self.state_elapsed), self.state_elapsed)
        # Fresh idle duration for envs cycling back -- sample a full-shape
        # tensor and select via the mask (torch.where needs matching shapes,
        # not a mask-sized tensor scattered in).
        fresh_idle_duration = torch.empty_like(self.idle_duration).uniform_(*self.cfg.idle_time_range)
        self.idle_duration = torch.where(to_idle, fresh_idle_duration, self.idle_duration)
        self._launch_yaw_captured = torch.where(
            to_idle, torch.zeros_like(self._launch_yaw_captured), self._launch_yaw_captured
        )
        self._crouch_entry_captured = torch.where(
            to_idle, torch.zeros_like(self._crouch_entry_captured), self._crouch_entry_captured
        )

        self.crouch_active = self.state == 1
        self.launch_active = self.state == 2
        active = (self.crouch_active | self.launch_active).float()

        self.phase = torch.where(
            self.crouch_active,
            (self.state_elapsed / self.cfg.crouch_duration).clamp(0.0, 1.0),
            torch.where(self.launch_active, torch.ones_like(self.state_elapsed), torch.zeros_like(self.state_elapsed)),
        )
        # Forward-only direction (same v5 staging decision as the old jump
        # line: 100% of training pressure on the one direction that requires
        # genuine 4-leg launch work). dir_x=active, dir_y=0, phase in slot 2 --
        # identical 3-slot layout to every other velocity-family command in
        # this codebase, so warm-start dimension-compatibility holds.
        self._command[:, 0] = active
        self._command[:, 1] = 0.0
        self._command[:, 2] = self.phase

        # Capture launch_yaw exactly once, the first step this cycle enters
        # launch -- see this attribute's own __init__ comment.
        entering_launch = self.launch_active & ~self._launch_yaw_captured
        if torch.any(entering_launch):
            q = self.robot.data.root_quat_w
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            yaw_now = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            self.launch_yaw = torch.where(entering_launch, yaw_now, self.launch_yaw)
            self._launch_yaw_captured = self._launch_yaw_captured | entering_launch

        # Capture crouch_entry_thigh/calf exactly once, the first step this
        # cycle enters crouch -- see this attribute's own __init__ comment.
        entering_crouch = self.crouch_active & ~self._crouch_entry_captured
        if torch.any(entering_crouch):
            thigh_now = self.robot.data.joint_pos[:, self._thigh_joint_ids]
            calf_now = self.robot.data.joint_pos[:, self._calf_joint_ids]
            mask = entering_crouch.unsqueeze(-1)
            self.crouch_entry_thigh = torch.where(mask, thigh_now, self.crouch_entry_thigh)
            self.crouch_entry_calf = torch.where(mask, calf_now, self.crouch_entry_calf)
            self._crouch_entry_captured = self._crouch_entry_captured | entering_crouch


@configclass
class JumpCrouchLaunchCommandCfg(CommandTermCfg):
    class_type: type = JumpCrouchLaunchCommand
    resampling_time_range: tuple[float, float] = (0.75, 1.5)  # nominal only, see _resample_command
    asset_name: str = "robot"
    # (2.0, 4.0) -> (0.75, 1.5) (2026-08-27, base's diagnosis, empirically
    # confirmed): mean episode length (~127 steps/2.54s) sat BELOW the old
    # idle_time_range's own mean (3.0s/150 steps) -- most episodes were dying
    # mid-idle, before crouch even completed, so launch_active_ratio measured
    # EXACTLY 0.0000 across the entire run's log (6440/6440 samples). Idle
    # standing is a warm-started skill the donor already has (nothing left to
    # train there); spending 60-70% of every episode's death-risk budget on
    # an already-solved phase starved the training distribution of any
    # launch samples at all, so no launch-phase reward term could ever fire a
    # gradient regardless of its weight. Verified with a direct stochastic-
    # noise probe (jump_v10_stochastic_idle_probe.py): injecting the
    # checkpoint's own logged noise_std=1.7 through the TRAINING action_scale
    # (0.5, not the bench's smaller per-joint convention) reliably produced
    # 1000s-of-Newtons illegal-contact-scope events on base_link/hip/thigh
    # across every tested seed, including one strike at t=2.50s while still
    # in idle state. Shortening idle reduces total exposure time to this
    # noise before crouch/launch can be reached. b2_jump_v10_driver.py's own
    # bench-side idle duration updated to match (same reasoning as every
    # other driver/config sync in this file's history).
    idle_time_range: tuple[float, float] = (0.75, 1.5)
    crouch_duration: float = 1.5
    # Same calibrated value as the old jump_env_cfg.py's FLIGHT_CLEARANCE_EPS
    # (MuJoCo-verified ~0.027m standing residual, ~1.85x margin) -- reused
    # directly rather than re-deriving, same physical quantity.
    liftoff_clearance_eps: float = 0.05
    launch_min_before_idle: float = 0.2  # seconds held in launch after genuine liftoff
    launch_hard_cap: float = 0.8  # seconds, safety net if liftoff never fires this cycle
    debug_vis: bool = False


# CORRECTED 2026-08-26 ~22:xx: the owner sent a frame-by-frame joint-angle
# table measured directly off the reference video (train_research/
# JUMP_V10_VIDEO_JOINT_REFERENCE.md) -- the 0.30m guess below this comment's
# own predecessor used was almost exactly 2x too SHALLOW. FK on the video
# table's LIE_ON_TERRA phase gives 0.150m (mean over 4 legs), closing the
# open item the original comment already flagged ("may want a fold deeper
# than 0.30m"). Averaged (not per-leg) across the 4 legs -- the table's own
# ~0.1rad per-leg spread is inside the owner's stated +-3-5deg measurement
# tolerance, and an intentionally asymmetric target would fight the owner's
# own repeated "ровно" (level/symmetric) requirement. See the reference
# file's own "Averaged reward targets" section for the full reasoning.
CROUCH_THIGH_TARGET = 1.5965  # rad -- point anchor, see jump_v10_crouch_pose
# CROUCH_CALF_TARGET (-2.7353, the raw video average) RETIRED 2026-08-27 as a
# point-anchor target -- see jump_v10_crouch_pose's own 2026-08-27 comment for
# why (0.085rad from the joint's own physical limit, a whole hour of training
# produced zero movement at weight -14). CROUCH_CALF_MIN_FLEX replaces it as a
# one-sided hinge threshold: -2.6 is comfortably inside the video's own
# observed range (-2.658..-2.756, plus RR at the limit -2.82) while leaving
# real headroom before the physical stop -- folding calf MORE than this
# (up to and including the joint's own limit) costs nothing, only
# under-folding is penalized.
CROUCH_CALF_MIN_FLEX = -2.6  # rad
CROUCH_TARGET_HEIGHT = 0.150  # rad-derived FK height, kept for docs/gate use only -- reward now anchors joint pose directly, not height

# NOTE: the crouch-pose ramp's phase=0 anchor (added 2026-08-28, base's
# design) originally used FIXED constants here (B2's nominal standing pose,
# unitree.py::UNITREE_B2_CFG -- thigh=0.8rad, calf=-1.5rad). Replaced
# 2026-08-29 (base's design) with a per-env MEASURED anchor
# (term.crouch_entry_thigh/calf, captured at the actual idle->crouch
# transition) after idle's own achieved pose drifted from these constants
# one night into training -- see jump_v10_crouch_pose's own "MEASURED
# CROUCH-ENTRY ANCHOR" comment for the full story. No constants needed here
# anymore; kept as a note, not dead values, so the history isn't lost.

# Ballistic launch velocity target -- see module docstring for the h=v^2/2g
# derivation (owner's 30-40cm clearance target -> 2.43-2.80 m/s). Clamp ceiling
# set above the target (not AT it) so exceeding the minimum bar keeps paying,
# same "cap discourages nothing, just removes a runaway single-term blowup"
# reasoning the old file's own jump_vertical_launch docstring already
# established.
JUMP_V10_LAUNCH_VZ_CLAMP = 3.5
# Forward component (owner's mid-design correction: "толчок вперёд И вверх",
# reference frame 2 "на пол-корпуса вперёд" at the same clearance the vertical
# target derives from). No independent ballistic derivation for this one --
# "half a body length" horizontal carry over the short liftoff instant this
# term actually gates (not the whole flight) is a much softer target than the
# vertical one; clamp kept at the same order of magnitude as v_z pending a
# real measurement once the vertical component is confirmed working.
JUMP_V10_LAUNCH_VX_CLAMP = 3.0


def _feet_off_ground_count(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.linalg.norm(forces, dim=-1) > 1.0
    return (~in_contact).float().sum(dim=1)


# Left-right joint pairs, front-vs-rear groups -- identical to the old
# jump_env_cfg.py's own _LR_JOINT_PAIRS/_FRONT_THIGH_CALF/_REAR_THIGH_CALF
# (same physical robot, same asymmetry-of-interest joints). Duplicated here
# (not imported) so this file has zero coupling to the old file's reward
# functions -- only FOOT_GEOM_LOCAL_OFFSET (a pure geometry constant, not a
# reward-shaping choice) is shared.
_LR_JOINT_PAIRS = [
    ("FL_hip_joint", "FR_hip_joint"),
    ("FL_thigh_joint", "FR_thigh_joint"),
    ("FL_calf_joint", "FR_calf_joint"),
    ("RL_hip_joint", "RR_hip_joint"),
    ("RL_thigh_joint", "RR_thigh_joint"),
    ("RL_calf_joint", "RR_calf_joint"),
]
_THIGH_JOINTS = ("FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint")
_CALF_JOINTS = ("FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint")


def _lr_asymmetry(asset) -> torch.Tensor:
    joint_names = asset.joint_names
    left_ids = [joint_names.index(left) for left, right in _LR_JOINT_PAIRS]
    right_ids = [joint_names.index(right) for left, right in _LR_JOINT_PAIRS]
    return torch.sum(torch.abs(asset.data.joint_pos[:, left_ids] - asset.data.joint_pos[:, right_ids]), dim=1)


def jump_v10_crouch_pose(env, command_name: str) -> torch.Tensor:
    """CORRECTED 2026-08-26 (owner's video-measured joint reference, see
    train_research/JUMP_V10_VIDEO_JOINT_REFERENCE.md): replaces the old
    root-height-only jump_v10_crouch_depth. A scalar base_z target says
    nothing about LEG SHAPE (and is itself somewhat fakeable by body pitch,
    the same class of issue the old jump line's FLIGHT_CLEARANCE_EPS
    redesign already fought once) -- anchoring thigh+calf directly to the
    owner's own measured crouch pose is both more precise (it IS the video
    frame, not a guess) and non-gameable (joint angles, not a root-frame
    quantity). Same shared symmetric target for all 4 legs -- see the
    reference file's own reasoning for why the table's small per-leg spread
    is treated as measurement noise, not an intentional asymmetry to learn.
    Hip excluded: the video table shows hip essentially frozen across all 4
    phases of the whole jump, nothing to anchor beyond what stand_still/
    joint_pos_penalty already cover.

    CORRECTED AGAIN 2026-08-27 (base's diagnosis, calf half): a point
    anchor at CROUCH_CALF_TARGET (-2.7353) sits only 0.085rad from the
    calf joint's own physical limit (-2.82, confirmed directly in
    urdf/b2.xml -- same for all 4 legs, not a hardware asymmetry). A whole
    hour of training (~2200 iterations) at weight -14 produced ZERO
    measured movement in the achieved calf angle -- the point-anchor
    was fighting something a squared-error weight alone couldn't win.
    The video reference itself doesn't ask for a precise -2.7353 anyway:
    RR_calf in the raw table IS the limit (-2.82) -- the real dog folds
    calf to its own limit too. Calf is now a ONE-SIDED (hinge) penalty:
    zero cost for folding AT LEAST to CROUCH_CALF_MIN_FLEX, penalty only
    for under-folding (calf less flexed / less negative than the
    threshold). This stops fighting the physical limit instead of trying
    to out-weight it, and matches what the video actually shows. Thigh
    stays a point anchor -- its target isn't limit-adjacent and it's the
    joint that actually defines the crouch's shape.

    RAMPED TARGET, STEP -> LINEAR-IN-PHASE (2026-08-28, base's diagnosis
    + design): root cause of the 2+ hour launch_active_ratio=0.0000
    plateau after the crouch_descent_rate fix (weight -1.75) -- a STEP
    target (full depth demanded from crouch's very first step) pays off
    EVERY step spent already at depth, so reaching it as fast as possible
    was always the reward-maximizing move; a single anti-speed penalty
    can never out-argue an anchor built to reward speed (-14 vs -1.75 is
    not a close fight, and it isn't a fixable weight ratio -- ANY step
    target creates this race by construction). Fix: interpolate the
    target BY PHASE instead of demanding it instantly. `term.phase` is
    already computed 0->1 across the crouch window by JumpCrouchLaunch
    Command and already sits in the command observation (slot 2) -- the
    policy already has direct access to it. Being at the ramped target
    every step is now BY DEFINITION a controlled descent; falling faster
    than the ramp is punished by the anchor itself (below-ramp = pose
    error), not just by the separate vz term -- the two terms are now
    allies, not opponents, closing the exploit class (arrive early by
    ANY means) rather than chasing this particular fast-fall shape of it.
    Calf's one-sided hinge interpolates its THRESHOLD (not a point target,
    per its own hinge design) -- at phase=0 the threshold equals B2's own
    standing calf angle (trivially satisfied, zero pressure to fold
    early), ramping to the full CROUCH_CALF_MIN_FLEX by phase=1.

    MEASURED CROUCH-ENTRY ANCHOR, NOT AN ASSUMED CONSTANT (2026-08-29,
    base's diagnosis + design): the phase=0 end of the ramp above used
    fixed STAND_THIGH_TARGET/STAND_CALF_TARGET constants (B2's nominal
    standing pose) -- found to drift out of sync one night into training,
    when idle's own achieved pose shifted (calf -1.5->-1.0rad, base_z
    0.54->0.635m) without the ramp's own anchor updating, reintroducing
    exactly the kind of phase=0 discontinuity/pressure the ramp was built
    to eliminate. Fix: capture the REAL joint angles at the actual
    idle->crouch transition per-env (term.crouch_entry_thigh/calf, latched
    once by JumpCrouchLaunchCommand itself, same idiom as launch_yaw) and
    ramp FROM there instead of from an assumed constant. phase=0 is now
    ALWAYS pressure-free by construction, regardless of whatever idle
    equilibrium the policy currently holds -- this decouples crouch-
    shaping from idle-drift permanently, not just until the next shift."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    joint_names = asset.joint_names
    thigh_ids = [joint_names.index(n) for n in _THIGH_JOINTS]
    calf_ids = [joint_names.index(n) for n in _CALF_JOINTS]
    phase = term.phase.unsqueeze(-1)
    thigh_target = term.crouch_entry_thigh + phase * (CROUCH_THIGH_TARGET - term.crouch_entry_thigh)
    calf_min_flex = term.crouch_entry_calf + phase * (CROUCH_CALF_MIN_FLEX - term.crouch_entry_calf)
    thigh_err = torch.sum(torch.square(asset.data.joint_pos[:, thigh_ids] - thigh_target), dim=1)
    calf_underflex = (asset.data.joint_pos[:, calf_ids] - calf_min_flex).clamp(min=0.0)
    calf_err = torch.sum(torch.square(calf_underflex), dim=1)
    return (thigh_err + calf_err) * term.crouch_active.float()


def jump_v10_crouch_feet_planted(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Feet must stay planted and still through the fold -- no re-stepping,
    splaying, or slamming (same idiom/reasoning as the old jump_crouch_feet_
    planted: a real pre-jump fold just bends the legs, no foot ever moves)."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    off_ground = _feet_off_ground_count(env, sensor_cfg)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.linalg.norm(forces, dim=-1) > 1.0
    foot_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, 0:2]
    slip = (torch.linalg.norm(foot_vel, dim=-1).clamp(max=3.0) * in_contact.float()).sum(dim=1)
    return (off_ground + slip) * term.crouch_active.float()


def jump_v10_level(env, command_name: str, phase: str) -> torch.Tensor:
    """Owner's core constraint, both phases: 'корпус ровно, без тангажа/крена'.
    projected_gravity_b xy is exactly zero when the body is level, regardless
    of yaw -- gravity-projected tilt, not a chosen orientation anchor, same
    non-gameable quantity the old file's jump_body_level already used.

    launch-phase gate EXTENDED 2026-08-29 (base's design, step (а) of the
    somersault fix -- see step (б), jump_v10_illegal_contact's landing-grace
    sibling in terminations, and self.all_airborne's own new docstring in
    JumpCrouchLaunchCommand): the old `launch_active` gate turned off the
    instant LAUNCH_DURATION's own timer expired (0.8s), while the body was
    still airborne on the kick's own momentum for a second+ longer -- exactly
    the window the 2026-08-29 live-bench somersault happened in, with ZERO
    orientation pressure the whole time. `term.all_airborne` is ground-truth
    "not touching the ground right now" (contact-sensor derived, updated every
    step), so `launch_active | all_airborne` covers the real inertial-flight
    tail regardless of which phase-timer label the state machine happens to
    be in at that instant -- this closes the gap at its source; mdp.
    bad_orientation (step б, same day) remains the hard safety backstop
    underneath it, not replaced by this."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    g_xy = asset.data.projected_gravity_b[:, 0:2]
    tilt = torch.sum(torch.square(g_xy), dim=1)
    gate = term.crouch_active if phase == "crouch" else (term.launch_active | term.all_airborne)
    return tilt * gate.float()


def jump_v10_yaw_rate(env, command_name: str) -> torch.Tensor:
    """Owner's core constraint: 'без YAW-верчения' -- priced directly as yaw
    angular velocity squared (not a heading-drift anchor), active through
    BOTH crouch and launch (the whole active cycle, not just the push).

    Gate EXTENDED 2026-08-29 alongside jump_v10_level -- same reasoning, same
    all_airborne addition, same somersault-window gap this closes."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    active = (term.crouch_active | term.launch_active | term.all_airborne).float()
    return torch.square(asset.data.root_ang_vel_w[:, 2]) * active


def jump_v10_crouch_descent_rate(env, command_name: str) -> torch.Tensor:
    """NEW 2026-08-28 (base's diagnosis + design): direct fix for the real
    mechanism behind today's hours-long launch_active_ratio=0.0000 plateau,
    found via jump_v10_illegal_contact_trace.py + a root vz probe -- the
    body free-falls into the crouch fold (root vz measured at -1.7 to
    -1.74 m/s deterministically, 6/6 identical trials) and slams base_link
    into the ground at 1884.9N when the fall is arrested, well past
    illegal_contact's 5N threshold and its own grace window. Every joint
    in this same window looked reasonable on all 4 legs (not the lagging-
    leg pattern chased all day) -- crouch_pose only ever anchored the
    FINAL joint angles, never the speed of getting there, so PPO had zero
    gradient pressure against a fast/violent fold.

    This is the stock mdp.lin_vel_z_l2 term (root vz squared) -- retired
    globally for jump_v10 (self.rewards.lin_vel_z_l2.weight = 0, this
    file's own retirement list) specifically because launch legitimately
    NEEDS vz, so a global penalty would fight the whole point of this
    skill. Reviving it here, gated to crouch_active ONLY, restores the
    penalty exactly where it's wanted without touching launch. Symmetric
    square, not one-sided -- crouch should have no fast vertical motion in
    EITHER direction (a fast upward rebound would just be the old jump
    line's own turnaround exploit in a new shape).

    Deliberately NOT an impact-force penalty -- illegal_contact already
    punishes the actual collision (terminates the episode outright, the
    strongest possible signal); this term is preventive (shapes the
    approach so the collision stops happening at all), not a second
    punitive layer on the same event."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    return torch.square(asset.data.root_lin_vel_w[:, 2]) * term.crouch_active.float()


def jump_v10_lr_symmetry(env, command_name: str, phase: str) -> torch.Tensor:
    """L-R joint symmetry, registered for crouch and launch separately (same
    quantity, different gate/weight -- launch symmetry matters more, an
    asymmetric push is a one-leg kick, not a jump)."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    gate = term.crouch_active if phase == "crouch" else term.launch_active
    return _lr_asymmetry(asset) * gate.float()


def jump_v10_launch_thigh_hold(env, command_name: str) -> torch.Tensor:
    """NEW 2026-08-26 (owner's video reference + explicit instruction:
    'сильный толчок всеми 4 НИЖНИМИ суставами (и только ими!), всё
    остальное неподвижно... верхние суставы держат'). The video table
    confirms this numerically: thigh moves ~0.04rad between LIE_ON_TERRA and
    JUMP_BEGIN (inside the owner's own +-3-5deg measurement tolerance --
    i.e. genuinely not moving) while calf moves ~0.56rad over the same
    interval -- see JUMP_V10_VIDEO_JOINT_REFERENCE.md.

    Prices raw THIGH joint velocity during launch (base's proposed design,
    reviewed): simpler than capturing-and-holding a reference angle (no new
    per-cycle capture state needed, one less thing that can drift out of
    sync), and 'не вращаются' (don't rotate) is literally a velocity
    statement, not a position one. Calf is deliberately NOT constrained here
    -- it is exactly the joint the physics-honest v_z/v_x outcome reward
    (jump_v10_launch_vertical/forward) is meant to drive; this term supplies
    the STRUCTURE (which joint does the work) the outcome reward alone
    cannot specify."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    joint_names = asset.joint_names
    thigh_ids = [joint_names.index(n) for n in _THIGH_JOINTS]
    thigh_vel_sq = torch.sum(torch.square(asset.data.joint_vel[:, thigh_ids]), dim=1)
    return thigh_vel_sq * term.launch_active.float()


def jump_v10_launch_vertical(env, command_name: str) -> torch.Tensor:
    """Main launch objective, component 1/2: upward velocity. min-over-4-hip
    v_z (NOT root v_z) -- root v_z is fakeable by pitching nose-up (rearing)
    without any leg genuinely pushing; during a rear/pitch cheat the rear hips
    barely rise, so min-over-4-hips collapses to ~0 and kills the credit at
    its source. Same non-gameable quantity + reasoning as the old file's
    jump_vertical_launch. Clamped to JUMP_V10_LAUNCH_VZ_CLAMP, comfortably
    above the 2.43-2.80 m/s physical target (see module docstring) so clearing
    the target doesn't hit a ceiling."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    hip_v_z = asset.data.body_lin_vel_w[:, term._hip_body_ids, 2]
    min_hip_v_z = hip_v_z.min(dim=1).values
    return min_hip_v_z.clamp(0.0, JUMP_V10_LAUNCH_VZ_CLAMP) * term.launch_active.float()


def jump_v10_launch_forward(env, command_name: str) -> torch.Tensor:
    """Main launch objective, component 2/2: forward velocity, ADDED by the
    owner's mid-design correction ('толчок вперёд И вверх', reference frame 2:
    'на пол-корпуса вперёд' at the same clearance the vertical target
    derives from -- the two components are meant to be comparable, not one
    dominating).

    Rotates world-frame root velocity into the yaw FIXED at launch-start
    (term.launch_yaw), never the live/current yaw -- this is the exact fix
    the old jump line had to add AFTER measuring a real exploit (v7d
    postmortem: a policy that yaws mid-push turned world-frame lateral drift
    into free 'forward' credit once a forward-velocity term existed). Adding
    this term from day one with the fix already in place, not after finding
    the same exploit again.

    CORRECTED 2026-08-27 (owner's live bench verdict on it22300: "прыгает
    НАЗАД" -- jumps BACKWARD): the original clamp(0.0, ...) floor made
    backward velocity pay exactly ZERO, not negative -- a free ride, not a
    penalty. The B2's own mechanics make a rear-dominant push drift
    backward NATURALLY (documented at length in the old jump line's own
    v5 forward-only staging note: "the hind-leg-dominant push drifts the
    body backward NATURALLY... backward is likely the robot's EASIEST
    direction"). With backward priced at zero and forward's own ignition
    not yet established, the policy had no gradient pressure against the
    mechanically-easier backward direction. Symmetric clamp now prices
    backward as an active penalty of the same magnitude as the forward
    reward, closing the free-ride."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    cos_yaw, sin_yaw = torch.cos(term.launch_yaw), torch.sin(term.launch_yaw)
    vel_w = asset.data.root_lin_vel_w[:, 0:2]
    vel_forward = vel_w[:, 0] * cos_yaw + vel_w[:, 1] * sin_yaw
    return vel_forward.clamp(-JUMP_V10_LAUNCH_VX_CLAMP, JUMP_V10_LAUNCH_VX_CLAMP) * term.launch_active.float()


# Clamp for the ignition scaffold below -- same order of magnitude as the calf's
# own rated speed isn't needed here (this rewards the DIRECTION of extension,
# not raw speed for its own sake); picked to keep a single leg's runaway
# velocity from dominating the sum, same "bound every term's worst case"
# discipline this file already follows elsewhere.
JUMP_V10_CALF_EXTEND_CLAMP = 3.0  # rad/s


def jump_v10_launch_calf_extend(env, command_name: str) -> torch.Tensor:
    """IGNITION SCAFFOLD (2026-08-27, base's diagnosis + design, reviewed):
    min-over-4-hip v_z (jump_v10_launch_vertical) is the right non-gameable
    OUTCOME target, but it is also, structurally, an anti-ignition trap --
    identical math to the old jump line's own jump_flight bootstrap problem
    and to WALK's feet_air_time gap: a min-across-4-legs quantity pays ZERO
    for any PARTIAL attempt (one or two legs, a timid half-push), so a
    policy that has never yet produced a coordinated 4-leg push gets no
    gradient toward discovering one -- while the risk side of the economy
    (tilt -6, lr_symmetry -6, thigh_hold -8) prices even a clumsy attempt.
    Freezing is the only locally-safe strategy until ignition happens by
    sheer exploration luck, which never occurred in a full night of
    training under this exact economy (jump_v10_launch_vertical/forward
    stayed flat ~0.002-0.005 the whole run -- see TRAINING_STATE.md
    2026-08-27 postmortem).

    This term is the owner's own described mechanism ("сильный толчок
    нижними суставами") translated directly into a reward, not a new
    invention: reward CALF EXTENSION VELOCITY per-leg (calf angles are
    negative at rest, deeper in the crouch; extension = becoming LESS
    negative = positive joint velocity), summed over all 4 legs, clamped.
    Unlike jump_v10_launch_vertical this is NOT gated on coordination
    across legs -- a single leg's first tentative extension attempt pays
    immediately, giving a gradient from step one instead of requiring a
    synchronized 4-leg event to discover. Non-gameable in the same way the
    file's other raw-joint terms are: extending the calf from a deep,
    feet-planted crouch physically raises the body (there is no cheap
    alternative way to move a calf joint that fakes this), and any
    lever-style workaround (pitching to fake something) is already priced
    by the existing tilt/lr_symmetry terms, not left free here.

    Marked SCAFFOLD deliberately (same discipline as the old jump line's
    own jump_rear_feet_liftoff/jump_crouch_depth_at_transition): once real
    ignition is established and jump_v10_launch_vertical/forward carry
    their own weight, revisit whether this term should shrink or drop out
    -- it exists to solve a bootstrap problem, not as a permanent shaping
    choice.

    CONTACT-GATED PER-LEG (2026-08-29, base's diagnosis + design): a real
    exploit, not a hypothesis -- jump_v10_calf_extend_vs_contact.py traced
    per-leg calf velocity alongside per-leg ground contact through the
    launch window and caught it directly: FL's calf velocity is a modest
    +1.9..+2.5rad/s while planted (a real push under load), then the
    INSTANT that leg goes airborne it spikes to +5.1..+9.09rad/s -- free
    windmilling, no load, no thrust, pure reward-farming (this scaffold
    paid raw extension SPEED with no contact requirement at all). FR shows
    the identical signature moments later (+1.9..+4.17 planted -> +7.09..
    +7.60 airborne). This is exactly why launch_calf_extend kept climbing
    (0.10->0.15) while launch_vertical stayed flat ~0.012 -- the policy
    was farming the scaffold's own blind spot instead of learning genuine
    thrust, and the rear legs (still grounded, still paying the modest
    honest rate) never got a reason to catch up.

    Fix: gate each leg's own extension credit to ONLY when that SAME leg
    is in ground contact -- extension under load is the actual invariant
    ("толчок", not "разгибание") the owner asked for, not a per-leg
    manual tuning knob. This closes the airborne-windmill exploit AND
    naturally shifts credit toward whichever legs are still doing real
    work (typically the rear pair, still planted while front is airborne)
    without guessing a front/rear weight ratio -- base's own reasoning for
    preferring this over an asymmetric-weight scaffold as the first move."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    joint_names = asset.joint_names
    calf_ids = [joint_names.index(n) for n in _CALF_JOINTS]
    extend_vel = asset.data.joint_vel[:, calf_ids].clamp(0.0, JUMP_V10_CALF_EXTEND_CLAMP)

    # Same ground-truth contact query + FL/FR/RL/RR ordering the command
    # term already uses for its own liftoff detection (_update_command) --
    # reused directly, not reimplemented, so this can never drift out of
    # sync with what "in contact" means elsewhere in this same command.
    contact_sensor = env.scene.sensors[term._feet_sensor_cfg.name]
    in_contact = (contact_sensor.data.current_contact_time[:, term._feet_sensor_order] > 0.0).float()

    return (extend_vel * in_contact).sum(dim=1) * term.launch_active.float()


# Knee (calf BODY origin, not the foot geom) height below which the shin is
# resting on the ground, not held up by a genuine crouch -- see
# jump_v10_no_lying_down's own docstring for the FK calibration. Legitimate
# poses (even the deepest allowed, calf at its own physical limit) sit at
# knee_z=0.130m; the failure mode this catches (thigh~1.24, calf~-2.826,
# base_link+all-4-calves confirmed bearing ground-contact force directly via
# scratchpad/jump_v10_check_ground_contact.py) sits at knee_z=-0.006m --
# comfortable margin on both sides.
JUMP_V10_MIN_KNEE_HEIGHT = 0.08  # m
# 2026-08-28 (base's diagnosis): re-verified directly against the 3 healthy
# legs of an actual trained checkpoint (it49900, jump_v10_all_leg_knee_
# trace.py) rather than the original FK estimate alone -- healthy legs sit
# at knee_z=0.13-0.30m in a genuine crouch, comfortable margin above 0.08,
# confirming the THRESHOLD itself was never the problem. Only a single
# lagging leg dipped to ~0.03m, continuously -- see jump_v10_no_lying_
# down's own "QUORUM REDESIGN" docstring for why single-leg triggering
# (not this threshold value) was the actual bug.
JUMP_V10_LYING_DOWN_QUORUM_LEGS = 2  # legs simultaneously below threshold to terminate
JUMP_V10_LYING_DOWN_QUORUM_DEBOUNCE_STEPS = 12  # consecutive steps quorum must hold (base's 10-15 range)

# 2026-08-29 (base's diagnosis + design, see jump_v10_illegal_contact's own
# "LANDING GRACE ADDED" comment): steps of illegal_contact immunity right
# after launch_active, covering the landing-impact transient so a genuine
# liftoff attempt doesn't get punished by its own landing -- same order of
# magnitude as the crouch/launch quorum debounce above (15 steps = 0.3s,
# slightly longer since a landing settle plausibly takes a touch more than
# the crouch-fold's own transient dip). Not a tuned final value -- a
# stopgap sized to stop the immediate impact spike, not to engineer a
# genuinely soft landing (out of scope this pass).
JUMP_V10_LANDING_GRACE_STEPS = 15
# Settle window after episode reset before jump_v10_no_lying_down starts
# checking -- see that function's own "BUG FOUND AND FIXED" comment.
# 100 -> 30 (2026-08-27 night): sized to fit under the new shortened idle
# (see JumpCrouchLaunchCommandCfg's own comment). WRONG -- overnight
# 15000-iteration segment (it30400-45400) still showed launch_active_
# ratio=0.0000 for the ENTIRE segment, episode length settled to a WORSE
# ~44-48 (vs ~65-100 before this change), jump_v10_no_lying_down still
# dominant (~88 vs illegal_contact ~0.7). My own settle-time probe (30
# steps) was measured against a probe that didn't replicate the OTHER
# reset-time randomization events (randomize_reset_base's own +-0.5 m/s
# / +-0.5 rad/s velocity kick, randomize_actuator_gains' 0.5x-2.0x
# stiffness/damping scaling, both mode="reset") -- so 30 was tuned
# against an artificially calm reset, undershooting the real (variable-
# duration) settle time. 30 -> 50 (2026-08-27 night, second pass): a
# reasoned middle value between the two data points now actually
# measured on real training (100 too long: wall exactly at grace,
# crouch_active_ratio=1.0 at the wall; 30 too short: no_lying_down
# dominant, episode length depressed below even the pre-idle-shortening
# baseline) -- not re-derived from a fresh clean-room estimate, since
# the previous one was already shown to under-model the real reset
# dynamics. Paired with narrowing the reset velocity kick itself (see
# randomize_reset_base.params["velocity_range"] override, same
# __post_init__, same reasoning) -- attacking both the settle-window
# SIZE and the shock MAGNITUDE that makes any fixed window a moving
# target. Expect to revisit this constant again once real data comes in;
# flagged explicitly rather than presented as solved.
JUMP_V10_LYING_DOWN_GRACE_STEPS = 50


def jump_v10_no_lying_down(env, command_name: str) -> torch.Tensor:
    """TERMINATION (2026-08-27, base's diagnosis + design): direct fix for a
    real physics exploit caught via ground-truth contact inspection, not a
    reward-shaping guess. jump_v10_crouch_pose (even at weight -14) never
    moved the achieved pose off thigh~1.24/calf~-2.826 -- base's hypothesis,
    confirmed by directly reading MuJoCo contact data during the crouch
    window: base_link AND all 4 calf bodies carry real ground-contact force
    (40.9N / 193.9N / 210.6N / 144.7N / 81.7+61.2N) at that pose. The robot
    is not holding a crouch, it is LYING DOWN -- passive ground support
    costs ~zero torque and is maximally stable under PPO's own exploration
    noise, which is why it beat every reward-weight escalation tried
    tonight (crouch_pose -8 then -14): a per-step reward penalty, no matter
    how large, is still just one more term in a sum a stable free-rest
    strategy can absorb, whereas a TERMINATION removes ALL future reward
    for the rest of the episode -- a categorically stronger, and cheaper to
    tune, deterrent (same economic-alignment idiom as the old jump line's
    own architectural gates, e.g. freezing time_left instead of fighting a
    reward race).

    Checks the CALF BODY's own world-Z (the knee, per FOOT_GEOM_LOCAL_
    OFFSET's own convention -- the calf body origin sits at the knee, the
    foot collision geom is a local offset from it) directly via ground-
    truth simulation state, NOT a contact-sensor body-level query --
    contact sensors in this MJCF resolve per BODY, and the foot geom is
    part of the calf body, so a sensor-based check cannot distinguish
    'foot legitimately touching the ground' (constant, expected) from 'shin
    resting on the ground' (the failure). Height is unambiguous: a foot
    touching the ground while the knee stays up is normal stance/crouch: a
    knee at or below ground level is only possible if the shin itself has
    gone flat.

    Not phase-gated -- there is no legitimate reason for a knee this low in
    ANY state of this skill (idle, crouch, launch, or the settle after
    launch), so this applies for the whole episode, same as any other
    catastrophic-failure termination in this codebase.

    BUG FOUND AND FIXED same session (first live check after enabling this):
    missing a spawn-settle grace period tanked mean episode length to
    ~18 steps (vs the normal ~1000) -- rough_env_cfg.py's own
    randomize_reset_base event spawns each episode at a fully random
    roll/pitch (+-pi) and up to +0.2m height (same spawn-fall transient
    documented at length in the old jump line's own min_foot_clearance
    calibration comment); an upside-down or tipped-over spawn instant has
    SOME body point near/below the knee threshold almost by construction,
    long before physics has time to settle into a real stance -- the
    termination was firing on spawn noise, not on the lying-down failure
    it was built to catch, and starving every episode of ever reaching
    crouch/launch (crouch_active_ratio collapsed to ~0.007, launch never
    even started). Fixed with a JUMP_V10_LYING_DOWN_GRACE_STEPS settle
    window via env.episode_length_buf, same idiom the old jump line's own
    JUMP_STANDING_CONSOLIDATION_STEPS uses for the same class of problem
    (though that one gates a whole command cycle, this just delays one
    termination check).

    QUORUM REDESIGN (2026-08-28, base's diagnosis + design, third pass at
    this exact termination): single-leg knee_z threshold turned out to
    kill episodes on a lagging leg's SUSTAINED (not transient) collapse --
    three grace-period retunings (100/30/50) all failed to unblock launch
    because grace only delays a death that was never a settling transient
    to begin with (jump_v10_all_leg_knee_trace.py confirmed the 3 healthy
    legs sit at knee_z=0.13-0.30m, comfortable margin -- only the ONE
    lagging leg sits at ~0.03m, continuously, for the whole crouch
    window). Raising crouch_lr_symmetry's weight or adding more grace both
    fail for the same reason: the reward/grace needs SURVIVED time in
    crouch to have any effect, and a single-leg-triggered termination
    guarantees death before that time is ever available -- a bootstrap
    deadlock, same species as the launch-ignition gap this file already
    solved once with jump_v10_launch_calf_extend.

    Fix: require TWO OR MORE legs below JUMP_V10_MIN_KNEE_HEIGHT
    simultaneously to terminate, not any single leg. A real lying-down
    failure is inherently multi-leg (a robot cannot rest its weight on the
    ground via one knee alone -- confirmed by the original exploit's own
    contact data above: base_link AND all 4 calves bore ground force
    together) while a single lagging leg mid-training is not that failure
    mode at all, just an undertrained leg. illegal_contact (base_link/hip/
    thigh, below) is untouched and keeps independently guarding the actual
    passive-rest failure via contact force, not height -- this quorum
    relaxation does not weaken that guard. Debounced over
    JUMP_V10_LYING_DOWN_QUORUM_DEBOUNCE_STEPS consecutive steps (not just
    the instantaneous quorum count) so a brief double-dip during the
    dynamic crouch-fold impulse itself doesn't false-trigger -- cheap
    insurance, doesn't change the core logic.

    Net effect this is designed to produce: episodes with one lagging leg
    now SURVIVE crouch and reach launch (even a lopsided 3-good-leg launch
    attempt) instead of dying before either the symmetry reward or the
    launch rewards ever get a survived step to train against -- this is
    expected to be what finally puts real launch samples into the
    training distribution, which crouch_lr_symmetry alone (added last
    cycle, confirmed active but starved of survival time) could not do by
    itself.

    PHASE-GATED QUORUM (2026-08-28, same night, found within the hour):
    the quorum-everywhere version above caused a real regression --
    jump_v10_idle_stability_scan on the very next checkpoint showed idle
    base_z std blown out from the usual 0.001-0.005 to 0.096 (range
    0.187-0.493m, should hover ~0.54m), confirmed visually
    (jump_v10_visual_correct_driver.py frames showing the robot flat on
    the ground during IDLE, not just crouch). Mechanism: the ORIGINAL
    single-leg threshold was doing TWO jobs at once -- catching the
    crouch-phase lagging-leg false-positive (the bug this whole redesign
    targets) AND acting as a general early-tip-over safety net during
    idle/launch (matching its own original "not phase-gated... no
    legitimate reason for a knee this low in ANY state" reasoning).
    Relaxing to a global 2-leg quorum fixed the first job but silently
    broke the second -- a robot starting to tip during idle now needs a
    SECOND leg to also dip low before anything catches it, so it can
    wobble/partially collapse for much longer before illegal_contact
    (a coarser, later-triggering, contact-based check) finally catches
    the actual fall. Fixed by gating the QUORUM to crouch_active only --
    idle and launch keep the original strict single-leg threshold (a low
    knee is still illegitimate in those phases, per the original
    reasoning, which turned out to be correct there -- it was only wrong
    for crouch specifically, where the video-reference target pose itself
    puts a healthy knee close to the ground).

    QUORUM EXTENDED TO LAUNCH (2026-08-28, same night, base's risk call --
    confirmed with data before applying, not applied preemptively): a
    lagging leg's knee does NOT heal instantly at the crouch->launch
    boundary -- jump_v10_all_leg_knee_trace.py on the live checkpoint
    (it52300) showed the lagging leg pinned at knee_z~0.030-0.031m FLAT
    across the entire crouch window, right up to the phase transition,
    zero sign of recovery. With the strict single-leg check active in
    launch, this guarantees instant death on launch's first step --
    death just relocates one phase later, exactly when launch samples
    are needed most, and launch_active_ratio would stay pinned near zero
    by the same underlying mechanism (matches the observed data this
    cycle: episode length capped ~55-58, launch_active_ratio still
    0.0000 across 30 samples). Extended the quorum+debounce condition to
    `crouch_active | launch_active` -- idle is now the ONLY phase with
    the strict instant single-leg check. Still safe: illegal_contact
    (base_link/hip/thigh, untouched) independently guards genuine
    lying-down in launch too, and is already the dominant terminator
    there; a single low knee during the dynamic launch push is either
    the lagging leg's own tail (crouch_lr_symmetry's job to fix, given
    time) or genuine extension-in-progress dynamics, not passive rest --
    idle is the one phase where legs are meant to be extended/standing,
    so a single low knee there is unambiguous."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    body_names = asset.body_names
    calf_body_ids = [body_names.index(f"{n}_calf") for n in ("FL", "FR", "RL", "RR")]
    knee_z = asset.data.body_pos_w[:, calf_body_ids, 2]
    legs_below = (knee_z < JUMP_V10_MIN_KNEE_HEIGHT).sum(dim=1)
    quorum_phase = term.crouch_active | term.launch_active
    required_legs = torch.where(
        quorum_phase,
        torch.full_like(legs_below, JUMP_V10_LYING_DOWN_QUORUM_LEGS),
        torch.ones_like(legs_below),
    )
    violation = legs_below >= required_legs

    if not hasattr(env, "_jump_v10_lying_down_streak"):
        env._jump_v10_lying_down_streak = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    fresh_episode = env.episode_length_buf <= 1
    env._jump_v10_lying_down_streak = torch.where(
        fresh_episode,
        torch.zeros_like(env._jump_v10_lying_down_streak),
        torch.where(
            violation,
            env._jump_v10_lying_down_streak + 1,
            torch.zeros_like(env._jump_v10_lying_down_streak),
        ),
    )
    # Debounce window follows the SAME crouch|launch gate as the quorum
    # itself (extended alongside it, same reasoning/data as the "QUORUM
    # EXTENDED TO LAUNCH" note above) -- only idle keeps an INSTANT
    # trigger (debounce requirement of 1 step), same as the original
    # pre-quorum design. Applying the 12-step cushion to idle was the
    # second half of the earlier idle-stability regression this phase-
    # gating fixed: a debounced single-leg check tolerates up to 12 steps
    # of an incipient idle tip-over before reacting, which is exactly the
    # kind of slow-motion collapse the original instant check existed to
    # prevent.
    required_streak = torch.where(
        quorum_phase,
        torch.full_like(legs_below, JUMP_V10_LYING_DOWN_QUORUM_DEBOUNCE_STEPS),
        torch.ones_like(legs_below),
    )
    debounced = env._jump_v10_lying_down_streak >= required_streak

    past_grace = env.episode_length_buf >= JUMP_V10_LYING_DOWN_GRACE_STEPS
    return debounced & past_grace


def jump_v10_illegal_contact(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Same as the stock mdp.illegal_contact, gated by the SAME spawn-settle
    grace period as jump_v10_no_lying_down (see that function's own "BUG
    FOUND AND FIXED" comment) -- the stock function is instantaneous, no
    grace concept at all, and randomize_reset_base's random-orientation
    spawn would trip base_link/hip/thigh contact almost every episode
    without this, the exact same failure class just found and fixed for
    the knee-height termination.

    LANDING GRACE ADDED (2026-08-29, base's diagnosis + design): with the
    2026-08-28 ramp fix, the policy started producing genuine partial
    liftoffs (front legs airborne, jump_v10_all_leg_knee_trace/visual
    confirmed) -- and landing back down now generates a real impact
    (1899-2064N on base_link, confirmed via jump_v10_illegal_contact_
    trace.py, right at the launch->idle boundary), which this instant
    check punishes exactly like the original crouch-entry crash. base's
    warning, confirmed by data across 4 consecutive hourly checks
    (illegal_contact aggregate climbing 0 -> 0.04 -> 0.08 -> 0.125,
    monotonic): "honest jump attempt -> hard landing -> termination ->
    'jumping = dying' economically" -- an anti-ignition trap that
    punishes exactly the behavior this whole redesign is trying to grow.
    Landing softness itself stays OUT OF SCOPE this pass (owner: "пока
    только это!") -- this grace does NOT train a soft landing, it only
    stops executing the policy for the landing impact transient right
    after a genuine launch attempt, same "punish the failure mode, not
    the attempt" idiom as the crouch/launch quorum's own debounce.
    JUMP_V10_LANDING_GRACE_STEPS steps of immunity start counting down
    from the last step launch_active was true (i.e. covers the landing
    impact immediately following launch, not indefinitely -- a genuine
    settle-then-lie-flat after landing will still eventually trip this
    once the grace window elapses)."""
    term = env.command_manager.get_term(command_name)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    illegal = torch.any(
        torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold, dim=1
    )
    past_grace = env.episode_length_buf >= JUMP_V10_LYING_DOWN_GRACE_STEPS

    if not hasattr(env, "_jump_v10_landing_grace"):
        env._jump_v10_landing_grace = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._jump_v10_landing_grace = torch.where(
        term.launch_active,
        torch.full_like(env._jump_v10_landing_grace, JUMP_V10_LANDING_GRACE_STEPS),
        torch.clamp(env._jump_v10_landing_grace - 1, min=0),
    )
    in_landing_grace = env._jump_v10_landing_grace > 0

    return illegal & past_grace & ~in_landing_grace


@configclass
class UnitreeB2JumpV10RoughEnvCfg(UnitreeB2RoughEnvCfg):
    """JUMP v10 -- minimal from-scratch crouch+launch only, see module docstring."""

    def __post_init__(self):
        super().__post_init__()

        # Flat ground -- same reasoning as the old jump line: the trick is
        # hard enough without rough terrain, and the standing-checkpoint
        # critic already tolerates a flat-constant height scan (see the old
        # file's own comment for why the scan itself is kept, not stripped).
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        # NEW 2026-08-27 (base's diagnosis): rough_env_cfg.py's own
        # randomize_reset_base spawns EVERY episode at a fully random roll/
        # pitch/yaw (+-3.14rad, confirmed by direct read of that file's own
        # params dict -- literally any orientation, including upside-down)
        # and up to +0.2m height. This is fine for the OLD jump-v3 line
        # (never had an illegal_contact-class termination -- floundering
        # through a tip-over recovery cost nothing but time) but is a real
        # problem now: jump_v10_no_lying_down + illegal_contact end the
        # episode on ANY base/hip/thigh ground contact, and self-righting
        # from a genuine upside-down spawn is a completely different skill
        # the stage_a_standing donor never learned -- explains the
        # deterministic episode-length wall at exactly the grace boundary
        # (100 steps, zero variance across hundreds of iterations) far more
        # cleanly than any reward-economy theory: a policy that is
        # PHYSICALLY UNABLE to right itself in time dies with 100% certainty
        # regardless of what it "wants" to do. Tumble-recovery robustness
        # is not this skill's job (it is not part of the owner's own task
        # description, which starts from a bench-confirmed standing pose,
        # not a fall) -- tamed to a near-standing spawn (small tilts, no
        # flips) so the illegal-contact terminations measure the actual
        # target skill (crouch/launch) instead of an unrelated recovery
        # skill nobody asked for. Mass randomization (randomize_rigid_body_
        # mass, a DIFFERENT event) is untouched -- that concern is orthogonal.
        self.events.randomize_reset_base.params["pose_range"]["roll"] = (-0.2, 0.2)
        self.events.randomize_reset_base.params["pose_range"]["pitch"] = (-0.2, 0.2)
        # yaw left at full range -- heading direction doesn't affect standing
        # balance at all, safe to randomize fully same as before.

        # NEW 2026-08-27 night (same reasoning as the pose narrowing above,
        # found AFTER a full 15000-iteration overnight segment still showed
        # launch_active_ratio=0.0000 with jump_v10_no_lying_down dominant --
        # narrowing orientation alone wasn't the whole story): the base
        # velocity_env_cfg.py's own randomize_reset_base ALSO kicks every
        # reset with up to +-0.5 m/s linear and +-0.5 rad/s angular velocity
        # on all 3 axes, on top of randomize_actuator_gains scaling
        # stiffness/damping 0.5x-2.0x at the SAME reset. This skill starts
        # from a bench-confirmed standing-at-rest pose (same donor as the
        # pose-range narrowing's own reasoning) -- a genuine tumble-recovery
        # velocity kick is not part of the task, and stacked with weak-gain
        # draws it can produce a real, variable-duration settle transient
        # my own spawn-settle probe (jump_v10_spawn_settle_probe.py, static
        # orientation offset only, no velocity/gain randomization) did not
        # replicate and therefore did not catch. Narrowed to a gentle nudge
        # rather than zeroed -- some reset variance is still useful (sim-to-
        # real robustness), just not violent enough to need multi-hundred-ms
        # recovery before the crouch/launch skill itself can even begin.
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (-0.1, 0.1),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.1, 0.1),
        }

        # DIAGNOSTIC A/B PROBE (2026-08-28, base's design) -- RUN AND
        # REFUTED, reverted to stock. Hypothesis: randomize_actuator_gains'
        # 0.5x-2.0x stiffness/damping draw at every reset was letting
        # "wattle leg" episodes sag knees below the quorum threshold on
        # legs that would be fine at nominal gains, explaining why every
        # direct bench check (always at default gains) looked healthy
        # while real training stayed stuck. Tested directly in the real
        # training distribution (narrowed to 0.9x-1.1x, ~287 iterations,
        # cheaper and more decisive than replicating IsaacLab's actuator
        # model on the bench): episode length stayed rock-flat ~55-57,
        # launch_active_ratio stayed EXACTLY 0.0000 on all 287 samples --
        # zero effect. Hypothesis refuted by data, not applied. Reverted
        # to stock range (kept commented for the record, not left active)
        # -- the mystery (launch_active_ratio pinned at 0 for hours despite
        # every direct knee-height check looking clean) remains open;
        # next candidate under investigation is illegal_contact (base_
        # link/hip/thigh), comparably dominant (34.1) but less directly
        # probed today than the knee-height/no_lying_down path.
        # self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.9, 1.1)
        # self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.9, 1.1)

        self.commands.base_velocity = JumpCrouchLaunchCommandCfg()
        # command_levels_vel reads `.cfg.ranges` off base_velocity -- this
        # command has none, the concept doesn't apply (same as the old line).
        self.curriculum.command_levels = None

        # Retire stock locomotion reward terms that assume a velocity-tracking
        # gait -- identical retirement list to the old jump_env_cfg.py, same
        # reasoning: this is a discrete trick, not a gait to track.
        self.rewards.track_lin_vel_xy_exp.weight = 0
        self.rewards.track_ang_vel_z_exp.weight = 0
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.feet_height_body.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_air_time.weight = 0

        # NEW 2026-08-26 ~23:5x (night-autonomous diagnosis, own initiative --
        # no direct owner order for this specific fix, standing "no blocking
        # questions" grant covers acting on a clear, measured regression):
        # idle-stability collateral drift found via jump_v10_idle_stability_scan.py
        # (std of settled-window base_z, t=1.5-3.0s of idle): it8000=0.012 (clean)
        # -> it8500=0.061 -> it8800=0.075 -> it9000=0.091 -> it9100=0.076
        # (all BEFORE the launch_thigh_hold escalation below) -> it9900=0.123 ->
        # it10700=0.141 (after). Monotonically worsening from the very start of
        # ordinary continued training under the crouch-pose-fixed economy, NOT
        # caused by the thigh_hold weight change specifically (that run started
        # from an ALREADY-drifting it9100, the escalation likely just added to
        # an existing trend, not the root cause). stand_still_without_cmd/
        # joint_pos_penalty (inherited from rough_env_cfg.py, -2.0/-1.0) are the
        # ONLY terms anchoring idle -- both gated on command norm < 0.1, so this
        # is a targeted, idle-only strengthening that cannot touch crouch/launch
        # economy at all. Doubled, same staged-escalation discipline as every
        # other weight change tonight.
        # -4.0/-2.0 -> -6.0/-3.0 (2026-08-27 ~05:35, night-autonomous, own
        # initiative): the SAME drift pattern started recurring, later and
        # slower this time -- jump_v10_idle_stability_scan.py std:
        # it19700=0.016 (normal) -> it19900=0.027 -> it20000=0.029, a real
        # jump within just 300 iterations, not a single-checkpoint fluke
        # (both post-jump points elevated vs the pre-jump one). The first
        # escalation (-2/-1 -> -4/-2) held the drift off for ~11000
        # iterations (it8000->it19700) but apparently wasn't a permanent
        # ceiling on the underlying pressure from the crouch/launch economy
        # -- doubling again, caught this time at std=0.029 instead of
        # waiting for it to reach 0.09+ like the first occurrence.
        self.rewards.stand_still_without_cmd.weight = -6.0
        self.rewards.joint_pos_penalty.weight = -3.0

        feet_sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_calf")

        # -- crouch (phase 1: "присела полностью") --
        # CORRECTED 2026-08-26: jump_v10_crouch_depth (root-height-only, target
        # 0.30m) REPLACED by jump_v10_crouch_pose (joint-space anchor to the
        # owner's video-measured reference, target height 0.150m -- see
        # JUMP_V10_VIDEO_JOINT_REFERENCE.md, CROUCH_THIGH_TARGET/CROUCH_
        # CALF_TARGET's own comment). This single pose anchor also subsumes
        # what jump_v10_crouch_lr_symmetry/jump_v10_crouch_front_rear_balance
        # (both REMOVED, not just re-weighted) used to approximate indirectly
        # -- a precise, video-sourced target for every leg makes those
        # free-variable proxies redundant, not just weaker.
        # -8.0 -> -14.0 (2026-08-27, owner caught it live: "0.078-0.09м --
        # по факту собака просто валяется на земле, так не должно быть").
        # Direct joint-angle trace (scratchpad/jump_v10_thigh_trace.py) on
        # the it20100 checkpoint confirmed it numerically, not just
        # visually: calf sits at -2.82..-2.83 (the joint's own PHYSICAL
        # LIMIT), not the target -2.7353 -- and RR_thigh sits at ~1.25 vs
        # ~1.5-1.65 on the other 3 legs, a real per-leg asymmetry the
        # target is supposed to prevent. -8.0 wasn't dominant enough
        # against the competing pull toward the joint limit (a mechanical
        # stop is a cheaper resting point for joint_torques_l2/joint_power
        # than precisely holding an intermediate angle against gravity) --
        # a hard floor-scraping pose leaves no calf travel left to actually
        # push from either, likely compounding the launch-ignition problem
        # on top of being visually wrong. Escalated, same staged discipline
        # as every other weight correction tonight -- not a new mechanism,
        # this term already targets the right numbers, it just needs to
        # win the argument against the joint-limit shortcut more decisively.
        self.rewards.jump_v10_crouch_pose = RewTerm(
            func=jump_v10_crouch_pose,
            weight=-14.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.jump_v10_crouch_feet_planted = RewTerm(
            func=jump_v10_crouch_feet_planted,
            weight=-5.0,  # matches the old jump_crouch_feet_planted
            params={"command_name": "base_velocity", "sensor_cfg": feet_sensor_cfg},
        )
        self.rewards.jump_v10_crouch_level = RewTerm(
            func=jump_v10_level,
            # -2.0 in the old line -> -4.0 here: owner named "no tilt" as a
            # near-hard constraint this pass ("следить за тем чтобы корпус
            # просто двигался ровно... всё"), one deliberate step up from the
            # old crouch-tilt weight, not a leap to the flight-phase value (-6.0).
            weight=-4.0,
            params={"command_name": "base_velocity", "phase": "crouch"},
        )
        # NEW 2026-08-28 (base's diagnosis + design, see jump_v10_crouch_
        # descent_rate's own docstring for the full mechanism): -1.75,
        # middle of base's suggested -1.5..-2.0 range -- at the measured
        # crash-causing vz=-1.7 m/s, this prices ~5.1/step, comparable
        # magnitude to the other crouch-economy terms (crouch_pose=-14
        # total across all 8 joint errors, crouch_level=-4) -- enough to
        # make a slow, controlled descent clearly cheaper than the
        # free-fall observed today, not so much that it fights descending
        # at all (the crouch still has to get down there).
        self.rewards.jump_v10_crouch_descent_rate = RewTerm(
            func=jump_v10_crouch_descent_rate,
            weight=-1.75,
            params={"command_name": "base_velocity"},
        )
        # NEW 2026-08-28 (bug found, not a design change): jump_v10_lr_
        # symmetry's own docstring says "registered for crouch and launch
        # separately" but only the launch half was ever actually wired in
        # below -- crouch had ZERO L-R symmetry pressure this whole time.
        # Found while chasing why launch_active_ratio stayed 0.0000 through
        # THREE different grace-period tunings (100/30/50): a direct knee-
        # height+joint trace (jump_v10_knee_vs_joint_trace.py) showed the
        # knee-height termination wasn't firing on a legitimate deep crouch
        # at all -- it was firing on the already-known RR/RL noisy-leg
        # collapse (thigh stuck near ~0.5-0.6rad, well under even the idle
        # stance angle, while calf over-folds to ~-2.4 to -2.6 on that one
        # leg only), which was dismissed earlier today as "not
        # catastrophic" before this consequence was known. Weight -4.0,
        # matching crouch_level's own "one step down from the launch-phase
        # equivalent" pattern (launch_lr_symmetry is -6.0) -- crouch
        # symmetry still matters less than launch's per that function's own
        # docstring, but zero was never the intended value.
        self.rewards.jump_v10_crouch_lr_symmetry = RewTerm(
            func=jump_v10_lr_symmetry,
            weight=-4.0,
            params={"command_name": "base_velocity", "phase": "crouch"},
        )

        # -- launch (phase 2: strong, symmetric, forward+upward push) --
        self.rewards.jump_v10_launch_vertical = RewTerm(
            func=jump_v10_launch_vertical,
            # Old line's single vertical term settled at 14.0; split roughly
            # in half with the new forward component per the owner's own
            # "примерно поровну" instruction -- adjust the split once real
            # data shows which component needs more pull.
            # 7.0 -> 10.0 (2026-08-27, paired with the thigh_hold -12->-8
            # rollback): ignition never happened all night (raw reward flat
            # ~0.002-0.005 the whole run) while thigh_hold was strong enough
            # to make "attempt nothing" the safe strategy -- raising the
            # payoff for a genuine push alongside lowering the risk of
            # attempting one, not just one or the other.
            weight=10.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.jump_v10_launch_forward = RewTerm(
            func=jump_v10_launch_forward,
            weight=10.0,
            params={"command_name": "base_velocity"},
        )
        # NEW 2026-08-27 (base's diagnosis + design -- see jump_v10_launch_
        # calf_extend's own docstring for the full ignition-gap analysis).
        # Weight 4.0: base's suggested range (3-5) -- below the main launch
        # outcome terms (10 each) so it never outranks genuine v_z/v_x once
        # those ignite, above pure noise so it actually moves the needle
        # during the current dead zone. SCAFFOLD -- revisit after ignition.
        self.rewards.jump_v10_launch_calf_extend = RewTerm(
            func=jump_v10_launch_calf_extend,
            weight=4.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.jump_v10_launch_level = RewTerm(
            func=jump_v10_level,
            weight=-6.0,  # matches the old line's dive-fix-calibrated flight-phase level weight
            params={"command_name": "base_velocity", "phase": "launch"},
        )
        self.rewards.jump_v10_launch_lr_symmetry = RewTerm(
            func=jump_v10_lr_symmetry,
            weight=-6.0,  # matches the old line's dive-fix-calibrated flight_symmetry weight
            params={"command_name": "base_velocity", "phase": "launch"},
        )
        # NEW 2026-08-26 (owner's video reference + explicit instruction --
        # see jump_v10_launch_thigh_hold's own docstring). This is a
        # DIFFERENT axis from lr_symmetry above: symmetry says "both sides
        # move the same amount", this says "the thigh shouldn't move AT ALL"
        # -- a push that's symmetric but uses the wrong joint would satisfy
        # lr_symmetry while still violating what the video actually shows.
        # -6.0 -> -12.0 (2026-08-26 ~23:0x, owner's LIVE bench verdict on
        # it8000/test_jump_v10: "вижу что пытается толкаться ВСЕМИ лапами и
        # верхней и нижней частью, нужно только НИЖНЕЙ" -- direct, live
        # confirmation the -6.0 constraint was not yet dominant enough:
        # thigh contributing to the push also contributes to v_z/v_x
        # (weight 7 each), so a soft thigh-hold loses that trade. Doubled --
        # same staged-escalation discipline as every other weight change
        # tonight (proportionate step, not a jump to an extreme), not a new
        # term (the reward economy training WAS already visibly, if not yet
        # sufficiently, pulling the right direction -- crouch_pose/thigh_
        # hold both trending down hard between it8063 and it9000, this is a
        # magnitude correction on a term already pointed correctly).
        #
        # -12.0 -> -8.0 (2026-08-27, OVERCORRECTION diagnosed): the it22300
        # checkpoint (trained ~5500+ it under -12.0) failed live on the
        # owner's bench -- "провал, падает вниз, прыгает НАЗАД... потом
        # просто падает". Re-verified with the CORRECT deploy driver
        # (jump_v10_visual_correct_driver.py -- the whole night's own
        # visual_policy_check.py --skill jump checks used the WRONG one,
        # b2_jump_pulse.py's JumpPulseDriver, a different command timing
        # entirely; see TRAINING_STATE.md for the full methodology
        # postmortem): base_z stays FLAT at the crouch depth (~0.10m)
        # through the ENTIRE launch-active window -- the policy does
        # NOTHING during launch, no calf extension either -- then an
        # uncontrolled collapse happens the instant launch_active drops
        # back to idle and the thigh_hold/crouch_pose pressure lifts. -12.0
        # made "freeze completely" the dominant safe strategy: any thigh
        # reaction from attempting a real calf extension risks a large
        # penalty, while the small near-zero launch_vertical/forward
        # reward (never ignited, flat ~0.002-0.005 all nightlong -- the
        # HONEST signal I should have trusted over the visual artifact)
        # wasn't enough to outweigh that risk. Halved back toward the
        # original -6.0 rather than reverting outright -- the owner's
        # underlying complaint (both joints moving) was real and confirmed
        # numerically at -6.0, so going below it isn't warranted, but -12.0
        # overshot into full paralysis.
        self.rewards.jump_v10_launch_thigh_hold = RewTerm(
            func=jump_v10_launch_thigh_hold,
            weight=-8.0,
            params={"command_name": "base_velocity"},
        )

        # -- whole active cycle (crouch + launch): no yaw spin, ever --
        self.rewards.jump_v10_yaw_rate = RewTerm(
            func=jump_v10_yaw_rate,
            # No direct precedent in the old line (it never had a standalone
            # yaw-rate term) -- first guess, same order of magnitude as the
            # other symmetry/orientation terms here, calibrate after the
            # first checkpoint if yaw still drifts.
            weight=-3.0,
            params={"command_name": "base_velocity"},
        )

        # NEW 2026-08-27 (base's diagnosis + design -- see jump_v10_no_
        # lying_down's own docstring for the full ground-contact-confirmed
        # exploit diagnosis). Termination, not a reward penalty -- ending
        # the episode removes ALL future reward, the only lever that can
        # structurally outbid a passive, zero-effort resting strategy no
        # per-step weight escalation beat tonight.
        self.terminations.jump_v10_no_lying_down = DoneTerm(
            func=jump_v10_no_lying_down, params={"command_name": "base_velocity"}
        )

        # NEW 2026-08-27 (base's diagnosis, cheap regardless of exact
        # mechanism): stock is_alive reward -- flat +1.0 per step simply for
        # not having triggered ANY termination yet. This is the role stock
        # positive tracking rewards (track_lin_vel_xy_exp etc., zeroed in
        # this file's own retirement list) play in the normal locomotion
        # economy -- "being alive and doing the task" nets positive there.
        # jump_v10 never had an equivalent floor: correcting an earlier
        # claim (base's own, then mine) that ALL positive reward was zeroed
        # here -- upward (unweighted, NOT in the retirement list) is in fact
        # still active and dominant on the batch average (+0.269/step
        # measured this same session) -- but that average can mask
        # per-trajectory negativity for whichever envs ARE contorted/dying,
        # and a small unconditional floor directly removes any possible
        # incentive to end an episode early regardless of the exact
        # mechanism, cheap insurance layered on top of the spawn-taming fix
        # above (the two are independent, address different candidate
        # causes for the same symptom).
        self.rewards.is_alive = RewTerm(func=mdp.is_alive, weight=1.0)

        # ADDED same day, ~1h after the term above (base's diagnosis): knee_z
        # is a PROXY for "resting on the calves specifically" -- the policy
        # found the nearest available cheat that satisfies the proxy while
        # keeping the same free-support strategy: contort so the knee stays
        # above the 0.08m threshold while shifting weight onto the HIP/THIGH
        # instead. Confirmed directly (jump_v10_check_ground_contact.py, it
        # 21700): FR_hip carrying 247N of ground-contact force, thigh angles
        # sprawled 0.07..2.7rad across the 4 legs -- not a random collapse,
        # the SHAPE of evading the specific metric being checked. Classic
        # Goodhart's-law failure: optimizing the proxy (knee height) instead
        # of the actual invariant the task needs ("only the feet -- the calf
        # tip -- may bear weight; base/hip/thigh never").
        #
        # Fix: re-enable the STOCK illegal_contact termination (velocity_env_
        # cfg.py's own DoneTerm, disabled -> None in rough_env_cfg.py for
        # locomotion tasks where transient non-foot contact is tolerable) and
        # scope it to base_link + every hip + every thigh -- reusing the
        # existing mechanism rather than inventing a second one, per base's
        # explicit ask. Calf is deliberately excluded: the foot collision
        # geom IS part of the calf body (b2.xml has no separate foot body,
        # same convention FOOT_GEOM_LOCAL_OFFSET's own comment documents),
        # so calf ground contact is the normal, legitimate case this term
        # must NOT punish -- jump_v10_no_lying_down above is the one that
        # catches illegitimate calf (shin, not foot-tip) contact via height,
        # since a contact SENSOR can't distinguish foot-tip from shin within
        # the same body. Together the two terms close the whole class: any
        # passive support other than the feet ends the episode, whichever
        # body it comes from.
        # jump_v10_illegal_contact (own wrapper, not mdp.illegal_contact
        # directly) -- adds the same spawn-settle grace period as
        # jump_v10_no_lying_down; the stock function is instantaneous and
        # would fire on the random-orientation spawn transient almost every
        # episode otherwise, the exact bug just found and fixed above.
        self.terminations.illegal_contact = DoneTerm(
            func=jump_v10_illegal_contact,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=[self.base_link_name, ".*_hip", ".*_thigh"]
                ),
                "threshold": 5.0,
            },
        )

        # NEW 2026-08-29 ~14:40 (base's diagnosis + design, SAFETY-CRITICAL,
        # applied same-session per his explicit "не жди меня, это чистый
        # стоковый рецепт"): owner caught a live somersault on the bench
        # (test_jump_v10, it102900) that no existing check surfaced --
        # jump_v10_level/jump_v10_yaw_rate are gated to
        # crouch_active|launch_active, which ends at LAUNCH_DURATION=0.8s
        # while the body is still airborne on the launch kick's momentum
        # for another second+ with ZERO orientation pressure of any kind.
        # Direct quaternion trace confirmed: pitch -28 deg still inside the
        # launch window, then +78 deg pitch / 160 deg roll AFTER launch ends
        # (jump_v10_pitch_and_splay_trace.py, robot_stand scratchpad). No
        # orientation termination existed in this task at all -- only
        # knee-height quorum and contact-force, neither catches mid-air
        # rotation. mdp.bad_orientation is stock IsaacLab
        # (isaaclab/envs/mdp/terminations.py) -- angle between projected
        # gravity and vertical, catches pitch+roll together as one number,
        # never wired into ANY B2 task in this repo before now. Deliberately
        # UNGATED (whole episode, including the post-launch inertial-flight
        # window where the actual tumble happens -- the gap this whole fix
        # exists to close). limit_angle=65deg: base's call, a stock
        # "about to fall over" range -- not tuned to the measured -28 deg
        # (likely legitimate nose-dip during a real push) vs +78/160 deg
        # (unambiguous tumble); the two are far enough apart that a rough
        # threshold separates them safely without precise calibration.
        # Ungated (а)-side fix (extend jump_v10_level/yaw_rate's gate to the
        # command class's existing all_airborne signal, not just the launch
        # timer) is base's proposed follow-up, queued for after this
        # backstop is confirmed live -- this termination is the first,
        # weight-free safety layer per his explicit ordering.
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 1.134},  # ~65 deg
        )

        # Known codebase gotcha (already documented in vision_env_cfg.py's own
        # 2026 comment, hit again here the hard way): UnitreeB2RoughEnvCfg's
        # own disable_zero_weight_rewards() call is gated on the exact class
        # name, so it does NOT fire for this subclass -- every stock
        # zero-weight scaffolding term (e.g. wheel_vel_penalty, dormant for
        # non-wheeled robots, empty joint_names) stays a real RewTerm instead
        # of being turned into None, and the reward manager crashes trying to
        # resolve its empty joint_names regex against B2's actual joints.
        # Every other leaf env cfg in this family (walk/crawl/rear_stand/the
        # old jump line) repeats this same call with its own class name --
        # required here too, not optional boilerplate.
        if self.__class__.__name__ == "UnitreeB2JumpV10RoughEnvCfg":
            self.disable_zero_weight_rewards()
