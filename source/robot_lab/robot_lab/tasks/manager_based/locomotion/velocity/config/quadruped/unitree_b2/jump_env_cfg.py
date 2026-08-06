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
    directions: tuple = ((1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (-1.0, 0.0))
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
CROUCH_TARGET_HEIGHT = 0.20  # base height with the belly nearly on the ground


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
    vertical velocity mean a crash-landing, not a landing. stand_still_without_cmd
    (command is zeros here) simultaneously pulls the joints back to the default
    stance -- together: touch down, kill the motion, stand."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    ang = torch.sum(torch.square(asset.data.root_ang_vel_w[:, 0:2]), dim=1)
    lin_z = torch.square(asset.data.root_lin_vel_w[:, 2])
    return (ang + lin_z) * term.landing_active.float()


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
            weight=-0.5,
            params={"command_name": "base_velocity"},
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
        self.rewards.jump_landing_impact = RewTerm(
            func=jump_landing_impact,
            weight=-0.5,
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
        self.rewards.jump_vertical_launch = RewTerm(
            func=jump_vertical_launch,
            weight=8.0,
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
        self.rewards.jump_direction_precision = RewTerm(
            func=jump_direction_precision,
            weight=-1.5,
            params={"command_name": "base_velocity"},
        )

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeB2JumpRoughEnvCfg":
            self.disable_zero_weight_rewards()
