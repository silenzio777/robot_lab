# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Discrete triggered-jump variant (2026-08-04 redesign).

History, short version (full record: robot_stand's train_research/TRAIN_RESEARCH.md):
the first "jump" pass kept the walking velocity-command interface and tried to shape a
bound gait purely through rewards (feet_gait re-paired front-vs-rear, feet_air_time) --
gait synchrony converged, flight phase never appeared. Second pass diagnosed the 10s
constant-velocity command sampler and switched to short pulses -- still no jump: the
command SEMANTICS were unchanged, "track this velocity", and the policy kept answering
with (crawl-like) locomotion, which is the correct answer to that question. A jump is
not a velocity to track; it is a discrete, self-terminating trick: crouch -> launch ->
flight -> land -> stand. This file changes the question.

Design (user's own spec, 2026-08-04):
- Command is a TRIGGER PULSE, not a velocity: idle (command = zeros, robot must stand
  still) -> jump window ~1.1s (command = direction unit vector in slots 0:2, and the
  window PHASE 0->1 in slot 2 -- the policy always knows where it is inside the jump)
  -> landing settle window (command = zeros again) -> next idle. One pulse = exactly
  one jump. Directions: forward / left / right. Backward is deliberately EXCLUDED per
  explicit request (most unnatural, would slow the others down).
- Same 3 command slots as every velocity variant -> observation stays 45-dim ->
  warm-startable from the walking checkpoint (standing and pushing off are already
  known; only the trick is new). The previous warm-start "failure" was the command
  semantics, not warm-starting itself.
- Rewards swap velocity tracking for jump events: distance covered WHILE AIRBORNE
  along the commanded direction (the definition of a jump, not a run), a flat flight
  bonus, and a landing-settle penalty right after the window. stand_still_without_cmd
  (inherited) already handles "don't move unless told to".
- Flat terrain (plane) -- a discrete trick is hard enough without rough ground;
  terrain/command curricula both off (command curriculum would crash anyway: it
  reads `.cfg.ranges` off the command term, which a pulse command doesn't have).

Jump distance target: ~0.5m to start (real B2 manages ~1m; scale up later, e.g. by
raising the airborne-velocity clip / window length, once 0.5m lands reliably).
"""

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeB2RoughEnvCfg


class JumpPulseCommand(CommandTerm):
    """Per-env cycle: idle -> jump window -> landing settle -> (resample) idle ...

    The base CommandTerm's own resampling clock drives the cycle: _resample_command
    picks this cycle's idle duration + direction and overwrites `time_left` with the
    FULL cycle length (idle + window + landing), so the base class resamples exactly
    when a cycle completes. _update_command then derives "where inside the cycle am
    I" from how much of time_left remains -- no second clock to keep in sync.
    """

    cfg: "JumpPulseCommandCfg"

    def __init__(self, cfg: "JumpPulseCommandCfg", env) -> None:
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        n = self.num_envs
        self._command = torch.zeros(n, 3, device=self.device)
        self.direction = torch.zeros(n, 2, device=self.device)
        self.idle_duration = torch.zeros(n, device=self.device)
        self.cycle_duration = torch.zeros(n, device=self.device)
        self.window_active = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.landing_active = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.phase = torch.zeros(n, device=self.device)
        self._directions = torch.tensor(cfg.directions, dtype=torch.float, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        # Fraction of envs currently inside a jump window -- a cheap "is the pulse
        # machinery alive at all" signal in TensorBoard, nothing decision-driving.
        self.metrics["window_active_ratio"] = self.window_active.float()

    def _resample_command(self, env_ids):
        # Base class just wrote time_left from resampling_time_range -- overwrite it
        # with this cycle's true length so the next resample lands exactly at cycle end.
        idle = torch.empty(len(env_ids), device=self.device).uniform_(*self.cfg.idle_time_range)
        self.idle_duration[env_ids] = idle
        self.cycle_duration[env_ids] = idle + self.cfg.window_duration + self.cfg.landing_duration
        self.time_left[env_ids] = self.cycle_duration[env_ids]
        dir_idx = torch.randint(0, self._directions.shape[0], (len(env_ids),), device=self.device)
        self.direction[env_ids] = self._directions[dir_idx]

    def _update_command(self):
        elapsed = self.cycle_duration - self.time_left
        window_start = self.idle_duration
        window_end = self.idle_duration + self.cfg.window_duration
        self.window_active = (elapsed >= window_start) & (elapsed < window_end)
        self.landing_active = elapsed >= window_end
        self.phase = torch.where(
            self.window_active,
            ((elapsed - window_start) / self.cfg.window_duration).clamp(0.0, 1.0),
            torch.zeros_like(elapsed),
        )
        active = self.window_active.unsqueeze(-1).float()
        self._command[:, 0:2] = self.direction * active
        self._command[:, 2] = self.phase


@configclass
class JumpPulseCommandCfg(CommandTermCfg):
    class_type: type = JumpPulseCommand
    # Nominal only -- _resample_command overwrites time_left with the real cycle
    # length every resample; this just satisfies the base cfg contract.
    resampling_time_range: tuple[float, float] = (2.0, 4.0)
    asset_name: str = "robot"
    idle_time_range: tuple[float, float] = (1.0, 2.5)
    window_duration: float = 1.1
    landing_duration: float = 0.5
    # Forward / left / right / BACKWARD unit vectors (robot frame). Backward was
    # excluded 2026-08-04 as "most unnatural" -- then re-added 2026-08-06 at the
    # user's request, with the exclusion itself debunked by the bench: the
    # hind-leg-dominant push drifts the body backward NATURALLY (the overnight
    # checkpoint jumped backward on a forward command), so backward is likely the
    # robot's EASIEST direction, not its hardest.
    # v5 FORWARD-ONLY (2026-08-16, owner's staging order after the live verdict
    # on v4 24999: forward jumps lift only the FRONT feet -- the rear stay
    # planted -- while backward flies clean with all four airborne; "Убираем все
    # три! Делаем ТОЛЬКО прыжок вперед! Четкий, ровный высокий прыжок вперед!").
    # Backward is the robot's free-ride direction (the rear-dominant push drifts
    # the body backward by itself) and left/right precision was historically
    # weakest; with all four directions in one run the averaged flight metrics
    # let the easy backward mask the weak forward all night. Forward-only puts
    # 100% of training pressure on the one direction that requires genuine
    # 4-leg launch work. COMPATIBILITY PRESERVED (owner's explicit requirement):
    # the command layout stays [dir_x, dir_y, phase] and obs stays 45 -- this
    # tuple only changes what gets SAMPLED, so a forward-only checkpoint
    # warm-starts a later re-add of the other directions unchanged.
    directions: tuple = ((1.0, 0.0),)
    debug_vis: bool = False


# 4-phase jump structure inside the 1.1s window (2026-08-05, modeled on how a real
# Go2/B2 actually executes a jump, per the user's own frame-by-frame observation):
#   1. CROUCH  (phase 0.00-0.35): fold down low, belly toward the ground
#   2. LAUNCH  (phase 0.35-0.60): explosive extension, rear-dominant push
#   3. FLIGHT  (phase 0.35-1.00): all four feet off, carry the direction
#   4. LANDING (the 0.5s landing slice after the window): absorb on all fours
# The policy always knows the phase (command slot 2) -- these constants gate WHICH
# reward is live WHEN, so the crouch->launch ordering is shaped explicitly instead
# of hoping exploration discovers it (it didn't -- see the squat-pogo post-mortem).
CROUCH_PHASE_END = 0.35
# 0.20 -> 0.35 (2026-08-12, bench verdict on 34999: "сильно приседает" -- the
# belly-low fold reads as a violent wind-up, and the real B2 reference video
# jumps from a SOFT knee bend, not a full fold. 0.35 from a 0.53 stand is that
# soft spring-load; the launch terms carry the rest.)
# 0.35 -> 0.30 (2026-08-12 evening, jump v2 run stalled: flight pinned at
# exactly 0.000 through it8400 with vertical_launch creeping 0.32->0.35 and no
# acceleration -- the v1 lineage had flight ~0.55 by the same iteration. The
# soft bend cut the leg-extension stroke too far AND feet_planted removed the
# re-stepping wind-up, leaving no ignition path. 0.30 returns part of the
# stroke while staying far from the old belly-low 0.20; paired with
# vertical_launch 8->10 below. Applied under the user's standing grant
# ("останавливать, изменять, запускать по новой -- можешь сам").)
CROUCH_TARGET_HEIGHT = 0.30  # base height at the bottom of the soft pre-jump bend
# v3->v4 (2026-08-16, owner bench verdict on 24999: "еле-еле прыгает", flight_distance
# down ~30% vs the prior checkpoint 33399, 0.22->0.16, despite vertical_launch and
# direction_velocity holding steady). Root cause traced to jump_airborne_front_leg_pose
# (added the night before): its gate was `phase >= CROUCH_PHASE_END`, so it fired
# from 0.35 onward -- covering LAUNCH (0.35-0.60, the explosive extension itself),
# not just true airborne coasting/landing-prep. Anchoring the front legs to the
# STANDING default pose during the push penalized the very leg extension a real
# launch needs, teaching a timid push instead of fixing the intended bug (a leg
# frozen mid-fold through flight). LAUNCH_PHASE_END names the boundary this term
# should have used from the start -- gates jump_airborne_front_leg_pose only, not
# jump_airborne_leg_stillness (unchanged since 2026-08-09, coexisted fine with
# 33399's own good flight_distance, not a new variable in this regression).
LAUNCH_PHASE_END = 0.60

# v6 (2026-08-19, positive landing package -- see jump_landing_pose/
# jump_landing_height_stance/jump_landing_still_bonus's own docstrings for
# the full rationale). All three FIRST GUESSES, not empirically calibrated --
# picked sharp enough to demand a genuine return-to-stance (softer than this
# file's shared TRACKING_SIGMA=0.25 equivalent would barely discriminate a
# good landing from a mediocre one), not copied from Atanassov's own sigma
# (their units/error-summation differ from ours).
JUMP_LANDING_POSE_SIGMA = 0.1
JUMP_LANDING_HEIGHT_SIGMA = 0.02
JUMP_LANDING_STILL_SIGMA = 0.5


def _feet_airborne(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """True per env when ALL feet are out of contact (the flight phase)."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    return ~in_contact.any(dim=1)


def jump_flight(env, command_name: str, sensor_cfg: SceneEntityCfg, min_base_height: float) -> torch.Tensor:
    """Flat per-step bonus for being fully airborne INSIDE the jump window -- makes
    the very first accidental hop immediately rewarding, before any distance is
    covered (bootstrap term).

    min_base_height gate (2026-08-05): the first from-scratch run reward-hacked the
    ungated version -- it sat compressed at ~0.2m through idle and pogo-trembled in
    the windows (bench-measured: crouch 0.20, "peak" 0.39, zero displacement, never
    reaches standing height at all), collecting airborne ticks for micro-hops. A
    real jump carries the base ABOVE standing height (0.55); airborne ticks below
    min_base_height now pay nothing, so squat-pogo earns zero and the only path to
    the flight terms is an actual launch. Flat terrain -> world Z is ground-true."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    high = (asset.data.root_pos_w[:, 2] > min_base_height).float()
    post_crouch = (term.window_active & (term.phase >= CROUCH_PHASE_END)).float()
    return _feet_airborne(env, sensor_cfg).float() * high * post_crouch


def jump_flight_distance(
    env, command_name: str, sensor_cfg: SceneEntityCfg, max_vel: float, min_base_height: float
) -> torch.Tensor:
    """Velocity along the commanded direction WHILE AIRBORNE inside the window --
    integrates to "distance covered in flight", which is the actual definition of a
    jump (a run covers distance grounded and earns nothing here). Clipped at max_vel
    so ballistic distance, not raw launch violence, is what pays."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    # Same min_base_height gate as jump_flight -- see its docstring (anti-squat-pogo).
    high = (asset.data.root_pos_w[:, 2] > min_base_height).float()
    # Direction is commanded in the robot's yaw frame -- rotate the world-frame base
    # velocity into it via the root quat's yaw (cheap 2D rotation).
    quat = asset.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    vel_w = asset.data.root_lin_vel_w[:, 0:2]
    vel_local_x = vel_w[:, 0] * cos_yaw + vel_w[:, 1] * sin_yaw
    vel_local_y = -vel_w[:, 0] * sin_yaw + vel_w[:, 1] * cos_yaw
    vel_along = vel_local_x * term.direction[:, 0] + vel_local_y * term.direction[:, 1]
    # Same sign fix as jump_direction_velocity (2026-08-06): backward carry while
    # airborne now pays negative instead of free zero.
    reward = vel_along.clamp(-max_vel, max_vel) * _feet_airborne(env, sensor_cfg).float() * high
    post_crouch = (term.window_active & (term.phase >= CROUCH_PHASE_END)).float()
    return reward * post_crouch


def jump_vertical_launch(env, command_name: str) -> torch.Tensor:
    """Dense bootstrap: upward base velocity inside the jump window pays IMMEDIATELY,
    grounded or not. Added 2026-08-04 after the first v3 run proved the flight terms
    alone can't bootstrap: both are gated on ALL FOUR feet already being airborne, an
    event a standing policy never produces by exploration noise, so 6000 iterations
    passed with jump_flight pinned at exactly zero while the policy settled into
    "stand through the window" (~200 reward from stand-still/regularizers) and PPO's
    noise_std climbed back up hunting a gradient that wasn't there. Any upward push
    now climbs toward the real jump: harder push -> higher v_z -> eventually flight,
    where the flight-gated distance term takes over. Clamped so a real launch
    (v_z ~2 m/s) dominates the crumbs a trot's own bounce could collect."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    # Launch phase only -- paying v_z during the crouch slice would reward skipping
    # the fold (and the crouch term would fight it); each phase owns its incentive.
    in_launch = term.window_active & (term.phase >= CROUCH_PHASE_END)
    # Credit cap history: 3.0 -> 2.0 (rebalance) -> back to 3.0 (2026-08-05, run
    # 16-43-25 stalled): clearing the flight gate from a 0.2m crouch needs
    # v_z ~2.7-2.8 m/s, so a 2.0 cap left a DEAD ZONE in the ladder -- pushing
    # harder than 2.0 paid nothing until flight suddenly paid at the gate, and
    # vertical_launch flatlined at a third of the previous run's level. Capping
    # was the wrong anti-altitude lever anyway: it doesn't discourage height, it
    # just removes gradient; the direction-vs-height balance is the WEIGHTS' job
    # (direction 8 vs vertical 4).
    return asset.data.root_lin_vel_w[:, 2].clamp(0.0, 3.0) * in_launch.float()


def jump_crouch(env, command_name: str) -> torch.Tensor:
    """Phase 1: during the crouch slice of the window, penalize distance from the
    belly-low height -- the deep fold is what loads the jump (a real dog's prep is
    nearly lying flat). Squared error, same shape as the idle anchor but pulling the
    OPPOSITE way, and only inside its own phase slice."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    in_crouch = term.window_active & (term.phase < CROUCH_PHASE_END)
    err = torch.square(asset.data.root_pos_w[:, 2] - CROUCH_TARGET_HEIGHT)
    return err * in_crouch.float()


def jump_crouch_feet_planted(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Feet must stay PLANTED AND STILL through the crouch (added 2026-08-12,
    bench verdict on 34999: during the pre-jump fold the dog rapidly re-steps,
    SPLAYS the feet wide and slams them into the floor -- "ОЧЕНЬ сильно бьет по
    полу". The reference video jump starts from a plain stand: no foot ever
    moves, the legs just bend and spring).

    Root cause is a pricing hole: jump_crouch prices base HEIGHT during the
    fold and jump_idle_still prices BASE velocity, but what the feet do during
    the crouch was completely free -- lifting, re-stepping, splaying and
    slamming all cost nothing (the master free-variable lesson, instance N;
    landing_impact only fires in the landing slice). Priced two ways at once:
    (a) each foot OUT of contact during the crouch pays (you cannot re-step or
    slam a foot that never leaves the ground -- and a foot that never leaves
    its print cannot splay either), (b) horizontal foot speed while IN contact
    pays (no sliding the feet outward along the floor instead). Both bounded
    by construction (contact count <= 4, speed clamped) -- the
    unbounded-squared-term postmortems elsewhere in this file stay respected."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = torch.linalg.norm(forces, dim=-1) > 1.0  # [N,4]
    off_ground = (~in_contact).float().sum(dim=1)
    # Horizontal speed of feet that ARE in contact = sliding/shuffling.
    foot_vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, 0:2]
    slip = (torch.linalg.norm(foot_vel, dim=-1).clamp(max=3.0) * in_contact.float()).sum(dim=1)
    in_crouch = term.window_active & (term.phase < CROUCH_PHASE_END)
    return (off_ground + slip) * in_crouch.float()


def jump_launch_attitude(env, command_name: str) -> torch.Tensor:
    """Penalize roll/pitch (body tilting away from level) through the
    post-crouch window (added 2026-08-12, bench verdict on 34999: the backward
    jump lifts the rear high into the air -- "попу поднимает сильно" -- and
    barely lands on balance). The inherited `upward` reward (3.0) prices this
    too weakly against launch terms at 8: a hind-dominant push can buy a big
    pitch-up cheaply. Squared world-frame roll+pitch rate is NOT used --
    attitude itself is the problem, so price the gravity-projected tilt
    directly (same quantity flat-orientation terms use everywhere else).
    Active post-crouch through the window (launch + flight): a level body at
    takeoff lands level -- the trajectory is ballistic."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    # projected_gravity_b xy components are 0 when the body is level.
    g_xy = asset.data.projected_gravity_b[:, 0:2]
    tilt = torch.sum(torch.square(g_xy), dim=1)
    in_free = term.window_active & (term.phase >= CROUCH_PHASE_END)
    return tilt * in_free.float()


def jump_motor_speed_violation(env, asset_cfg=None) -> torch.Tensor:
    """Penalty on leg joint velocity exceeding its REAL hardware speed rating
    (hip/thigh 23 rad/s, calf 14 rad/s), everywhere (no phase gate -- both launch
    push-off and landing impact are where this happens).

    Added 2026-08-07 after a live bench test found the trained checkpoint
    (b2_jump_48499) reading 100-126% of real velocity on the bench's own
    motor-limit bars for every direction, worst on the calf: forward 126%,
    backward 118%, left/right 106%. Root-cause verified empirically (scripted
    bench reproduction, not guessed): TORQUE was already correctly clipped to
    100% by the DC-motor curve (bench and training use the identical curve --
    confirmed byte-for-byte, ctrl==actuator_force in the reproduction) -- the
    curve shapes available torque near the speed limit but does NOT hard-clamp
    velocity itself, so momentum from an explosive push-off or a stiff landing
    can carry the joint past its rated speed for free. Nothing in this file
    priced raw joint speed before now -- the master free-variable lesson,
    instance N: locomotion skills never approach this because their own reward
    shaping never asks for peak speed, but a discrete explosive jump's entire
    mechanism is peak instantaneous power, so it needs its own explicit price.

    Excess ratio clamped to [0, 2] (2026-08-07, run 21-55-47: reward exploded to
    -1.1M on isolated iterations, ~30x more often than the pre-fix baseline run's
    own rare -1000-to--3000 outliers -- traced to THIS term: unbounded, a single
    rare physics-glitch env with a momentary extreme joint velocity (contact
    catastrophe, not a real jump event) produced an astronomical squared penalty
    with no ceiling, wrecking that batch's value-function target. Every other
    term in this file already bounds its own worst case (velocity clamps, force
    thresholds) -- this one didn't, and a term meant to gently price a 100-126%
    overspeed has no business paying out as if speed were 10000%."""
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    joint_names = asset.joint_names
    leg_ids = [i for i, n in enumerate(joint_names) if n.endswith(("_hip_joint", "_thigh_joint", "_calf_joint"))]
    limits = torch.tensor(
        [14.0 if joint_names[i].endswith("_calf_joint") else 23.0 for i in leg_ids],
        device=asset.data.joint_vel.device,
    )
    leg_vel = asset.data.joint_vel[:, leg_ids]
    excess = (leg_vel.abs() / limits - 1.0).clamp(min=0.0, max=2.0)
    return torch.sum(torch.square(excess), dim=1)


def jump_airborne_leg_stillness(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalty on leg joint velocity while genuinely airborne (all 4 feet off the
    ground, post-crouch) -- added 2026-08-09 (user, bench: "дрыганье ногами в воздухе").

    Once the feet leave the ground the trajectory is already ballistic -- nothing the
    legs do mid-flight changes where the robot lands (jump_direction_precision prices
    the launch push, not the air). No existing term touches this window: jump_flight/
    jump_flight_distance/jump_vertical_launch all pay for BEING airborne, not for what
    the joints do while there, and jump_landing_impact/jump_landing_settle only fire
    after touchdown -- an unpriced free variable, the same lesson as every other gap in
    this file. Priced hard per the user's explicit request ("жестко штрафовать").

    Same normalize-by-rated-limit + clamp discipline as jump_motor_speed_violation
    (its own docstring has the postmortem: an unbounded squared term let one rare
    physics-glitch tick blow up a batch's value-function target) -- clamped before
    squaring so this can't repeat that failure."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    joint_names = asset.joint_names
    leg_ids = [i for i, n in enumerate(joint_names) if n.endswith(("_hip_joint", "_thigh_joint", "_calf_joint"))]
    limits = torch.tensor(
        [14.0 if joint_names[i].endswith("_calf_joint") else 23.0 for i in leg_ids],
        device=asset.data.joint_vel.device,
    )
    normalized = (asset.data.joint_vel[:, leg_ids] / limits).clamp(-3.0, 3.0)
    airborne = _feet_airborne(env, sensor_cfg)
    post_crouch = term.window_active & (term.phase >= CROUCH_PHASE_END)
    return torch.sum(torch.square(normalized), dim=1) * (airborne & post_crouch).float()


def jump_airborne_front_leg_pose(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """L1 penalty on FRONT leg (thigh+calf) deviation from default pose while
    genuinely airborne (post-crouch) -- added 2026-08-14, bench verdict on
    jump_33399: backward jumps tuck the front-right leg through the whole
    flight, forward/left/right don't (owner's exact words: "назад опять
    правую переднюю подгибает").

    Root cause (code-verified, not guessed): this module's own JumpPulseCommand
    docstring already notes backward is the robot's naturally EASIEST launch
    direction -- the hind-leg-dominant push drifts the body backward for free,
    so on a backward command the front legs do far less active push-work than
    they do fighting that natural bias on forward/left/right. A front leg that
    never had to work can end up passively mid-fold right at liftoff.
    jump_airborne_leg_stillness (2026-08-09) then makes things worse for THAT
    specific case: it penalizes JOINT VELOCITY while airborne, which makes
    freezing wherever the leg already is the cheapest option -- it rewards
    freezing at a bad fold exactly as much as freezing at a clean extended
    pose, since it has no notion of WHERE, only of not moving. Free variable,
    same class of gap as jump_crouch_feet_planted/jump_idle_symmetry
    elsewhere in this file.

    Front legs only, thigh+calf (not hip, not rear) -- same joint selection
    as rear_stand's own rear_stand_front_legs_tuck fix for an analogous
    front-leg-tremor bug. Rear legs deliberately excluded: they are mid-
    push-through at the moment of liftoff, anchoring them to the STANDING
    default would fight the very extension that makes the jump happen.
    Plain L1 (not squared/exp) -- bounded by joint range by construction,
    same unbounded-term discipline as every other term in this file, and it
    only needs to NUDGE the stillness term's chosen freeze-point, not fight
    it outright.

    v4 fix (2026-08-16): gate moved CROUCH_PHASE_END (0.35) -> LAUNCH_PHASE_END
    (0.60) -- see LAUNCH_PHASE_END's own comment for the full postmortem.
    The original gate fired from the START of LAUNCH (0.35-0.60, the explosive
    push itself), penalizing front-leg extension during the very push that
    makes a strong jump, not just a bad pose held through true flight/landing.
    Bench verdict on 24999 confirmed the cost: flight_distance -30% vs the
    prior checkpoint (33399) despite vertical_launch/direction_velocity
    holding steady -- distance/duration regressed, launch power didn't, which
    is exactly what over-anchoring the LAUNCH-phase leg would produce."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    joint_names = asset.joint_names
    ids = [
        i for i, n in enumerate(joint_names)
        if n in ("FL_thigh_joint", "FL_calf_joint", "FR_thigh_joint", "FR_calf_joint")
    ]  # fmt: skip
    err = torch.sum(torch.abs(asset.data.joint_pos[:, ids] - asset.data.default_joint_pos[:, ids]), dim=1)
    airborne = _feet_airborne(env, sensor_cfg)
    past_launch = term.window_active & (term.phase >= LAUNCH_PHASE_END)
    return err * (airborne & past_launch).float()


def jump_landing_impact(env, command_name: str, sensor_cfg: SceneEntityCfg, soft_threshold: float) -> torch.Tensor:
    """Phase 4: absorption. During the landing slice, penalize foot contact force
    beyond a soft threshold -- a stiff-legged slam spikes way past it, an absorbing
    landing spreads the impulse out. Normalized by the threshold so the weight reads
    as "penalty per multiple-of-soft-limit"."""
    term = env.command_manager.get_term(command_name)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1)
    excess = (forces - soft_threshold).clamp(min=0.0).sum(dim=1) / soft_threshold
    return excess * term.landing_active.float()


def jump_idle_still(env, command_name: str) -> torch.Tensor:
    """Penalty on horizontal base velocity + yaw rate everywhere EXCEPT the jump
    window. Added 2026-08-05 after the first working checkpoint (b2_jump_3100 --
    real jumps, good 4-foot landings) showed parasitic idle shuffling: slow
    backward-and-left drift while "standing". Root cause is a pricing hole:
    stand_still_without_cmd prices JOINT-pose deviation and jump_idle_height prices
    HEIGHT, but XY translation and turning in idle cost nothing -- and the training
    curve confirms the policy actively trades idle stillness for jump reward
    (stand_still_without_cmd degraded -0.36 -> -1.88 while the jump terms grew).
    Nothing self-corrects a free variable; price it."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    v_xy = torch.sum(torch.square(asset.data.root_lin_vel_w[:, 0:2]), dim=1)
    w_z = torch.square(asset.data.root_ang_vel_w[:, 2])
    # Gate extended 2026-08-05 (run 16-43-25 exploit, visible as this term's own
    # value degrading -0.26 -> -0.77 while direction_velocity grew): the CROUCH
    # slice was covered by NO motion penalty at all (this term stopped at the
    # window edge, direction_precision starts post-crouch) -- a free run-up during
    # the "fold" that then cashed in as launch-slice direction_velocity. The fold
    # must be stationary, like a real dog's: still everywhere EXCEPT the
    # launch/flight slice, which is the only part of the cycle meant to move.
    in_free = term.window_active & (term.phase >= CROUCH_PHASE_END)
    return (v_xy + w_z) * (~in_free).float()


def jump_idle_height(env, command_name: str, target_height: float) -> torch.Tensor:
    """L2 height anchor active everywhere EXCEPT inside the jump window: idle and
    landing must be a TALL stand. Exists because nothing else anchors height here --
    base_height_l2 is deliberately zeroed (a jump's own height must swing freely
    inside the window), and without any anchor the first from-scratch run idled in a
    ~0.2m squat (the same free-variable-slides-away failure the vision variant hit,
    see TRAIN_RESEARCH.md 2026-08-04: unanchored behavior drifts wherever the other
    rewards push it). Flat terrain -> plain world-Z target, no scan needed."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    err = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    # The landing slice is deliberately EXCLUDED too (2026-08-05, user question
    # caught this): absorbing a landing means deep leg flexion -- the base dips to
    # ~0.3-0.4m -- and anchoring height there directly fights jump_landing_impact's
    # softness incentive; the cheap way to satisfy the anchor would be a stiff-legged
    # slam. Idle proper is the only phase that must be a tall stand; the anchor
    # takes over pulling the robot back up when the NEXT idle begins.
    idle = ~term.window_active & ~term.landing_active
    return err * idle.float()


# Left-right joint pairs shared by the symmetry terms below.
_LR_JOINT_PAIRS = [
    ("FL_hip_joint", "FR_hip_joint"),
    ("FL_thigh_joint", "FR_thigh_joint"),
    ("FL_calf_joint", "FR_calf_joint"),
    ("RL_hip_joint", "RR_hip_joint"),
    ("RL_thigh_joint", "RR_thigh_joint"),
    ("RL_calf_joint", "RR_calf_joint"),
]


def _lr_asymmetry(asset) -> torch.Tensor:
    """Summed L1 left-right joint-angle asymmetry over _LR_JOINT_PAIRS."""
    joint_names = asset.joint_names
    left_ids = [joint_names.index(left) for left, right in _LR_JOINT_PAIRS]
    right_ids = [joint_names.index(right) for left, right in _LR_JOINT_PAIRS]
    return torch.sum(torch.abs(asset.data.joint_pos[:, left_ids] - asset.data.joint_pos[:, right_ids]), dim=1)


def jump_idle_symmetry(env, command_name: str) -> torch.Tensor:
    """L1 penalty on left-right joint asymmetry during idle (added 2026-08-09,
    bench: right leg visibly tucked under the body during idle -- an unstable
    tripod-like stance, "как будто болит" -- while the crouch/launch/landing
    poses stayed fine. Root-cause diagnosed empirically (headless MuJoCo probe,
    not guessed): checkpoints 71000/91000/91799 are all left-right symmetric in
    idle (FR_hip vs FL_hip within 0.1 rad); only the latest fine-tune
    (91799->104700, which added jump_airborne_leg_stillness and tightened
    jump_direction_precision) drifted asymmetric (FR_hip=0.44 vs FL_hip=-0.01).
    Neither new term touches this -- both gate on term.window_active, zero
    during idle -- so this is collateral drift from continued training (one
    shared network across all phases), not a direct reward-shaping cause.
    stand_still_without_cmd/joint_pos_penalty (inherited, active in idle) price
    total L1 deviation from default but not asymmetry specifically -- a
    lopsided-but-bounded-magnitude pose can satisfy that sum cheaply. This
    prices |left - right| per joint pair directly so idle can't settle into a
    lopsided stance regardless of what else drifts elsewhere."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    idle = ~term.window_active & ~term.landing_active
    return _lr_asymmetry(asset) * idle.float()


def jump_flight_symmetry(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """L1 penalty on left-right joint asymmetry during FLIGHT and LANDING (added
    2026-08-11, bench verdict on b2_jump_124699: the right front leg stays folded
    through flight and touchdown while the left one gets thrown upward -- the same
    chronic asymmetry jump_idle_symmetry already prices in idle, but that term
    deliberately gates OFF for the whole window+landing, so the air and the
    touchdown were left unpriced (free variable, instance N).
    jump_airborne_leg_stillness prices joint VELOCITY in flight, not pose -- a leg
    frozen in a folded position is perfectly "still" and passes it for free.

    Gate: genuinely airborne post-crouch, OR the landing slice. A symmetric tuck
    mid-flight satisfies this at zero cost (both sides fold together); only a
    lopsided pose pays. Sideways jumps launch off asymmetric GROUNDED pushes --
    those stay free (this fires only once airborne, where the trajectory is
    ballistic and an asymmetric pose serves nothing)."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    airborne = _feet_airborne(env, sensor_cfg) & term.window_active & (term.phase >= CROUCH_PHASE_END)
    active = airborne | term.landing_active
    return _lr_asymmetry(asset) * active.float()


def jump_direction_velocity(env, command_name: str, max_vel: float) -> torch.Tensor:
    """Dense DIRECTIONAL bootstrap (2026-08-05, after run 14-38-22 converged to a
    vertical jump-in-place): velocity along the commanded direction during the
    post-crouch window slice -- grounded or airborne. The economics before this
    term: vertical_launch (dense, big) + flight (pays for airtime) both REWARD
    putting the whole impulse into height, while the only directional term
    (flight_distance) was gated on already-moving-horizontally-while-airborne --
    the same chicken-and-egg the flight bootstrap solved for height, unsolved for
    direction, measured as launch=1.19 vs flight_distance=0.03: jumping straight up
    was simply the better trade. This term makes the horizontal push pay from the
    first grounded shove; flight_distance then takes over in the air. Gated
    post-crouch so horizontal drift during the fold doesn't score."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    q = asset.data.root_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    vel_w = asset.data.root_lin_vel_w[:, 0:2]
    vel_local_x = vel_w[:, 0] * cos_yaw + vel_w[:, 1] * sin_yaw
    vel_local_y = -vel_w[:, 0] * sin_yaw + vel_w[:, 1] * cos_yaw
    vel_along = vel_local_x * term.direction[:, 0] + vel_local_y * term.direction[:, 1]
    in_launch = term.window_active & (term.phase >= CROUCH_PHASE_END)
    # clamp(0,max) -> clamp(-max,max) (2026-08-06, bench verdict on the overnight
    # checkpoint: "прыгает не вперёд, а чуть назад по команде вперёд"): zeroing the
    # negative projection made jumping BACKWARD free -- the hind-leg-dominant
    # vertical push naturally drifts the body backward, and nothing opposed it
    # (third instance of the unpriced-free-variable lesson). Negative projection
    # now SUBTRACTS at full weight: going backward against the command is the most
    # expensive thing a launch can do.
    return vel_along.clamp(-max_vel, max_vel) * in_launch.float()


def jump_direction_precision(env, command_name: str) -> torch.Tensor:
    """Penalty on OFF-AXIS motion during the post-crouch slice: the velocity
    component PERPENDICULAR to the commanded direction, plus yaw rate. Added
    2026-08-05 (user: "чтобы в полёте не отклонялся, а летел куда нужно"): flight
    is ballistic -- the trajectory is fixed at takeoff, so precision must be priced
    at the push. The along-direction terms pay only the PROJECTION onto the
    command, so a jump "forward and hard sideways" still collected the forward
    share while the sideways smear and mid-air body twist cost nothing
    (jump_idle_still prices wz only OUTSIDE the window). Now: along pays,
    across costs, twisting costs."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    q = asset.data.root_quat_w
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
    vel_w = asset.data.root_lin_vel_w[:, 0:2]
    vel_local_x = vel_w[:, 0] * cos_yaw + vel_w[:, 1] * sin_yaw
    vel_local_y = -vel_w[:, 0] * sin_yaw + vel_w[:, 1] * cos_yaw
    # Perpendicular component: rotate the local velocity into the command frame --
    # dir is a unit vector, so this is the plain 2D cross product.
    v_perp = vel_local_x * (-term.direction[:, 1]) + vel_local_y * term.direction[:, 0]
    w_z = asset.data.root_ang_vel_w[:, 2]
    in_launch = term.window_active & (term.phase >= CROUCH_PHASE_END)
    return (torch.square(v_perp) + torch.square(w_z)) * in_launch.float()


def jump_landing_settle(env, command_name: str) -> torch.Tensor:
    """Penalty during the post-window landing slice: angular thrash and residual
    vertical/horizontal velocity mean a crash-landing, not a landing. stand_still_without_cmd
    (command is zeros here) simultaneously pulls the joints back to the default
    stance -- together: touch down, kill the motion, stand.

    ang widened to all 3 axes + lin_xy added (2026-08-08, bench: b2_jump_71000
    doesn't fall anymore but every direction rotates 30-35 deg on touchdown --
    right/left ALSO drift backward on landing, not just in flight): the old
    roll/pitch-only ang term and jump_idle_still's own wz/v_xy pricing both
    penalize RATE, not the accumulated heading/position error a fast impulsive
    touchdown spin/skid leaves behind after the rate itself decays back to zero
    -- by the time idle_still's broad, episode-averaged penalty registers
    anything, the robot has already spun/slid and stopped. This is the exact
    moment (landing_active, freshly strengthened to weight -1.5) where that
    impulse actually happens -- price it there directly instead of relying on
    the diffuse aftermath term to catch it."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    ang = torch.sum(torch.square(asset.data.root_ang_vel_w), dim=1)
    lin_xyz = torch.sum(torch.square(asset.data.root_lin_vel_w), dim=1)
    return (ang + lin_xyz) * term.landing_active.float()


# v6 (2026-08-19, "clean forward jump" task from the owner via claude-tg-base):
# the landing-quality economy above (jump_landing_settle/jump_landing_impact)
# is entirely NEGATIVE -- small penalties (-1.5/-2.0) for a bad landing, with
# NOTHING positively rewarding a GOOD one. Research found a working real-
# hardware jump recipe on the SAME stack (Atanassov et al., arXiv:2401.16337,
# legged_gym+rsl_rl, Go1 90cm forward jump, code open) whose landing economy
# is class-different: a DOMINANT POSITIVE package (change_of_contact=10,
# default_pose=12, base_height_stance=20, post_landing_pos/ori at a very
# sharp sigma) rather than small deterrents. Ported the CONCEPT, not their
# literal numbers (different overall reward economy, different sigma scale
# already established in this file) -- four new terms below, all gated on
# `landing_active` (the same phase flag jump_landing_settle/jump_landing_impact
# already use), all POSITIVE exp-kernel rewards that STACK ON TOP OF the
# existing penalties rather than replacing them (a bad landing still costs
# via settle/impact; a good one now ALSO earns, instead of just costing
# nothing).
def jump_landing_pose(env, command_name: str, asset_cfg=None) -> torch.Tensor:
    """Positive: exp-tracking of ALL leg joints back to the default stance
    during the landing slice -- Atanassov's own `default_pose`. Sharp-ish
    sigma (this file's shared TRACKING_SIGMA scale would be too soft for a
    "snap back to stance" signal) -- first guess, not empirically calibrated,
    expect revision same as every other sigma in this codebase's history."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    joint_names = asset.joint_names
    leg_ids = [i for i, n in enumerate(joint_names) if n.endswith(("_hip_joint", "_thigh_joint", "_calf_joint"))]
    err = torch.sum(torch.square(asset.data.joint_pos[:, leg_ids] - asset.data.default_joint_pos[:, leg_ids]), dim=1)
    return torch.exp(-err / JUMP_LANDING_POSE_SIGMA) * term.landing_active.float()


def jump_landing_height_stance(env, command_name: str, target_height: float) -> torch.Tensor:
    """Positive: exp-tracking of base height back to the standing target
    during the landing slice -- Atanassov's own `base_height_stance`.
    Complements jump_idle_height (which deliberately EXCLUDES landing_active,
    see its own docstring) by covering exactly the window that term skips."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    err = torch.square(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-err / JUMP_LANDING_HEIGHT_SIGMA) * term.landing_active.float()


def jump_landing_feet_planted(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Positive: all four feet in ground contact during the landing slice --
    Atanassov's own `change_of_contact` (they price the DERIVATIVE of contact
    state changing; this prices the STATE directly -- simpler, same effect:
    a foot that keeps leaving and retouching the ground during landing_active
    never holds full 4-contact for long, so a bouncy landing scores low on
    every step it's airborne, exactly where change_of_contact would also
    dock it)."""
    term = env.command_manager.get_term(command_name)
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    in_contact = (torch.linalg.norm(forces, dim=-1) > 1.0).float()
    all_four = in_contact.sum(dim=1) / 4.0
    return all_four * term.landing_active.float()


def jump_landing_still_bonus(env, command_name: str) -> torch.Tensor:
    """Positive companion to jump_landing_settle's existing penalty: exp-reward
    for near-zero residual base velocity (linear+angular) during the landing
    slice -- Atanassov's own `post_landing_pos`/`post_landing_ori` (very sharp
    sigma in their units; same "first guess, not their literal number" caveat
    as jump_landing_pose above). jump_landing_settle keeps penalizing a BAD
    landing (raw squared, unbounded upside for how bad); this adds a bounded
    positive signal for a GOOD one, so "landed clean" earns instead of merely
    not losing."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    ang = torch.sum(torch.square(asset.data.root_ang_vel_w), dim=1)
    lin = torch.sum(torch.square(asset.data.root_lin_vel_w), dim=1)
    return torch.exp(-(ang + lin) / JUMP_LANDING_STILL_SIGMA) * term.landing_active.float()


@configclass
class UnitreeB2JumpRoughEnvCfg(UnitreeB2RoughEnvCfg):
    """See module docstring -- discrete triggered jump, not a locomotion gait."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Flat ground -- the trick is hard enough without rough terrain; revisit
        # terrain later if 0.5m jumps land. Deliberately NOT the full
        # UnitreeB2FlatEnvCfg switch: the height_scanner and the CRITIC's height_scan
        # observation stay alive, because the walking checkpoint this run warm-starts
        # from (seed_from_walking_5000) was trained with the asymmetric actor-critic
        # -- critic input 235 (48 + 187 scan) -- and stripping the scan here shrank
        # the critic to 48, which made runner.load() fail with a critic.0.weight
        # size mismatch (hit live, 2026-08-04). On a plane the scan just reads flat
        # constants -- a consistent, harmless input the critic already knows.
        # (policy.height_scan is already None from the parent -- actor stays 45.)
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None

        # The pulse command replaces the velocity command wholesale (same 3 slots,
        # so the observation layout -- and warm-start compatibility -- is unchanged).
        self.commands.base_velocity = JumpPulseCommandCfg()
        # command_levels_vel reads `.cfg.ranges` off the base_velocity term -- a
        # pulse command has no ranges; the concept doesn't apply.
        self.curriculum.command_levels = None

        # -- retire the velocity-tracking recipe (the "it crawls instead" root cause)
        self.rewards.track_lin_vel_xy_exp.weight = 0
        self.rewards.track_ang_vel_z_exp.weight = 0
        # Vertical motion IS the trick -- don't suppress it...
        self.rewards.lin_vel_z_l2.weight = 0
        # ...and don't penalize feet tucking toward the body mid-flight (parent
        # weight -5.0 exists to keep a WALKING gait's feet under the body).
        self.rewards.feet_height_body.weight = 0
        # A jump's own height genuinely leaves any fixed target -- same reasoning
        # as the first jump attempt's docstring, kept.
        self.rewards.base_height_l2.weight = 0
        # Previous attempt's bound-gait shaping -- retired with the approach. The
        # jump terms below define the objective directly; no gait prior needed.
        self.rewards.feet_gait.weight = 0
        self.rewards.feet_air_time.weight = 0

        # -- the jump objective itself
        # 5.0 -> 8.0 (2026-08-06, same cascade-damping fix): a fatter airborne
        # prize so the occasional successful flight gets amplified hard.
        self.rewards.jump_flight = RewTerm(
            func=jump_flight,
            weight=8.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
                # 0.6 -> 0.55 (2026-08-05): gentler first rung -- just above the
                # 0.53 standing height still means a genuine jump, reachable at
                # v_z ~2.6 from the crouch instead of ~2.8.
                "min_base_height": 0.55,
            },
        )
        self.rewards.jump_flight_distance = RewTerm(
            func=jump_flight_distance,
            weight=8.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
                # 0.5m target over a ~1.1s window whose flight slice is a fraction
                # of it -- 2.0 m/s of credited airborne velocity is plenty for that
                # and caps the incentive to launch ballistically.
                "max_vel": 2.0,
                "min_base_height": 0.55,
            },
        )
        self.rewards.jump_landing_settle = RewTerm(
            func=jump_landing_settle,
            # -0.5 -> -1.5 (2026-08-07, bench: backward jump "падает почти всегда"):
            # backward is the robot's naturally strongest direction (hind-leg-dominant
            # push), so it carries the most momentum into landing -- absorption was
            # priced the same for every direction while the hardest case needed more.
            weight=-1.5,
            params={"command_name": "base_velocity"},
        )
        # v6 (2026-08-19): positive landing package, see the four functions' own
        # docstrings + JUMP_LANDING_*_SIGMA's own comment. Weighted to genuinely
        # DOMINATE the small existing landing penalties (settle -1.5, impact -2.0)
        # without exceeding the launch-phase anchors (vertical_launch 10) -- a
        # good landing should pay comparably to a good launch push, not less.
        self.rewards.jump_landing_pose = RewTerm(
            func=jump_landing_pose,
            weight=6.0,
            params={"command_name": "base_velocity", "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.jump_landing_height_stance = RewTerm(
            func=jump_landing_height_stance,
            weight=6.0,
            params={"command_name": "base_velocity", "target_height": 0.53},
        )
        self.rewards.jump_landing_feet_planted = RewTerm(
            func=jump_landing_feet_planted,
            weight=5.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            },
        )
        self.rewards.jump_landing_still_bonus = RewTerm(
            func=jump_landing_still_bonus,
            weight=5.0,
            params={"command_name": "base_velocity"},
        )
        # New 2026-08-07 (bench: torque/velocity/joint-range bars all red, 100-126%
        # of real hardware -- see jump_motor_speed_violation's own docstring for the
        # verified root cause). Weight moderate: this must discourage a dangerously
        # fast leg swing without killing the launch itself (vertical_launch/direction
        # terms still dominate at 8/4).
        self.rewards.jump_motor_speed_violation = RewTerm(
            func=jump_motor_speed_violation, weight=-2.0, params={"asset_cfg": SceneEntityCfg("robot")}
        )
        # New 2026-08-09 (user, bench: "дрыганье ногами в воздухе во время самого
        # прыжка" -- explicit ask to "жестко штрафовать" this). Weight matched roughly
        # to jump_motor_speed_violation's own -2.0 (same normalize-by-limit + clamp
        # scale, so the two penalties sit in comparable units) but pushed harder per
        # the user's explicit request for a hard stop, not a gentle nudge.
        self.rewards.jump_airborne_leg_stillness = RewTerm(
            func=jump_airborne_leg_stillness,
            weight=-3.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            },
        )
        # New 2026-08-14 (bench verdict on jump_33399: backward-only front-right
        # leg tuck through the whole flight -- see jump_airborne_front_leg_pose's
        # own docstring for the full root-cause chain). Weight moderate (-2.0,
        # same tier as jump_motor_speed_violation) -- a nudge toward WHERE to
        # freeze, deliberately weaker than jump_airborne_leg_stillness's own -3.0
        # so it doesn't reintroduce the flailing that term was built to stop.
        self.rewards.jump_airborne_front_leg_pose = RewTerm(
            func=jump_airborne_front_leg_pose,
            weight=-2.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            },
        )
        # Weight history: 1.0 in run 2026-08-04_22-14-17 -- too weak, the exact
        # stand-still optimum returned (vertical_launch flatlined at ~0.02-0.04,
        # noise_std collapsed 1.18->0.67): a vigorous push costs more in action_rate/
        # joint_acc/torque penalties than +v_z at weight 1.0 pays inside a ~7%-duty
        # window. 8.0 makes pushing unambiguously profitable; flight terms raised to
        # 5.0 alongside so the economics KEEP improving once actually airborne.
        self.rewards.jump_crouch = RewTerm(
            func=jump_crouch,
            weight=-8.0,
            params={"command_name": "base_velocity"},
        )
        # New 2026-08-12 (bench on 34999: stomping/splaying/re-stepping during
        # the fold -- see jump_crouch_feet_planted's own docstring). Weighted
        # strong (-5.0): the user's verdict called this the worst defect, and
        # the term is bounded so it can't repeat the blowup class.
        self.rewards.jump_crouch_feet_planted = RewTerm(
            func=jump_crouch_feet_planted,
            weight=-5.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            },
        )
        # New 2026-08-12 (bench on 34999: backward jump lifts the rear high,
        # lands barely on balance -- see jump_launch_attitude's own docstring).
        # Moderate weight: a jump needs SOME pitch freedom, this prices only
        # the gross tilt that made landings marginal.
        self.rewards.jump_launch_attitude = RewTerm(
            func=jump_launch_attitude,
            weight=-2.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.jump_landing_impact = RewTerm(
            func=jump_landing_impact,
            # -0.5 -> -2.0 (2026-08-07, same bench finding as landing_settle above):
            # weak relative to vertical_launch/flight at 8 -- a stiff-legged slam was
            # cheaper than it looked once the launch got this powerful (~1-1.5m jumps,
            # far past the original 0.5m target).
            weight=-2.0,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
                # ~60kg robot: static stand is ~150N/leg; a 4x spike per leg reads
                # as a slam. Soft threshold per foot, penalty per multiple over it.
                "soft_threshold": 500.0,
            },
        )
        self.rewards.jump_idle_still = RewTerm(
            func=jump_idle_still,
            weight=-3.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.jump_idle_height = RewTerm(
            func=jump_idle_height,
            weight=-8.0,
            params={"command_name": "base_velocity", "target_height": 0.53},
        )
        # New 2026-08-09 (bench: right leg tucked under body in idle -- see
        # jump_idle_symmetry's own docstring for the empirical diagnosis chain).
        # Weight matched to jump_idle_still (-3.0, same idle-only gate) -- assertive
        # since it never competes with the jump-phase economics (zero outside idle).
        self.rewards.jump_idle_symmetry = RewTerm(
            func=jump_idle_symmetry,
            weight=-3.0,
            params={"command_name": "base_velocity"},
        )
        # New 2026-08-11 (bench on 124699: right front folded in flight/landing,
        # left thrown up -- see jump_flight_symmetry's own docstring). Moderate
        # weight: a symmetric tuck costs zero, so this only bites lopsidedness;
        # kept below idle_symmetry's -3.0 because flight competes with the live
        # jump economics (direction/precision) in a way idle never does.
        self.rewards.jump_flight_symmetry = RewTerm(
            func=jump_flight_symmetry,
            weight=-1.5,
            params={
                "command_name": "base_velocity",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_calf"),
            },
        )
        # Weight history (the jump-vs-direction economics took three passes):
        # v1: vertical 8, direction absent -> perfect vertical jump IN PLACE.
        # v2: vertical 4, direction 8 -> run 18-10-50 found the mirror exploit: a
        #     grounded DASH in the launch slice (direction paid 8*2=16/s vs
        #     vertical's 4*3=12/s, easier and fall-free) -- vertical_launch
        #     flatlined at a third of its breakthrough level, flight never came.
        # v3 (current): vertical 8 (the ladder strength that actually broke through
        #     to flight), direction 4 as the tiebreaker -- a DIRECTED jump
        #     (v_z~2.6 + v_along~1.5) totals ~27/s vs ~24/s for a pure vertical
        #     one and ~8/s for a dash, so jumping wins overall and jumping WITH
        #     the command wins among jumps; flight_distance (8, airborne) widens
        #     the directed margin further.
        # 8.0 -> 10.0 (2026-08-12 evening, same stall as CROUCH_TARGET_HEIGHT's
        # own 0.35->0.30 note: with the stomping wind-up priced away, the pure
        # vertical push needs a steeper gradient to beat the stand-through-the-
        # window optimum from a standstill).
        self.rewards.jump_vertical_launch = RewTerm(
            func=jump_vertical_launch,
            weight=10.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.jump_direction_velocity = RewTerm(
            func=jump_direction_velocity,
            weight=4.0,
            params={"command_name": "base_velocity", "max_vel": 2.0},
        )
        # -2.0 -> -0.5 (2026-08-06, run 20-57-56: sparks without fire): early
        # directed jumps are inevitably sloppy, and at -2.0 the smear+yaw tax on
        # every attempt (~2-3/s) ate most of the direction reward (~6/s), damping
        # the amplification loop -- flight flickered at 0.01-0.03 for 2500+
        # iterations with no cascade (the vertical run, which had NO precision tax,
        # cascaded within ~1800 of ignition). Precision matters AFTER the jump
        # exists: keep a nudge now, re-tighten in a fine-tune pass post-cascade.
        # -0.5 -> -1.5 (2026-08-06 fine-tune): the ignition-era softness served its
        # purpose (cascade happened); bench shows left jumps twisting the body hard
        # left -- yaw/smear discipline comes back now that the jump itself exists.
        # -1.5 -> -2.5 (2026-08-09, user: "отклонение от курса... просто выровнять"):
        # an earlier pass already tried -2.0 (see history above, "sparks without fire"
        # -- direction pressure crushed the launch itself before it could ignite) but
        # that was BEFORE vertical_launch/flight/direction_velocity all grew to their
        # current strength and before landing_settle priced yaw too -- the jump's own
        # economics now dominate this term by a much wider margin than they did then,
        # so re-tightening is a fresh attempt, not a blind repeat of the old mistake.
        # If this reproduces the old symptom (jump amplitude collapsing, not just
        # straightening), the next step is a partial revert toward ~-2.0, not further
        # escalation.
        self.rewards.jump_direction_precision = RewTerm(
            func=jump_direction_precision,
            weight=-2.5,
            params={"command_name": "base_velocity"},
        )

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeB2JumpRoughEnvCfg":
            self.disable_zero_weight_rewards()
