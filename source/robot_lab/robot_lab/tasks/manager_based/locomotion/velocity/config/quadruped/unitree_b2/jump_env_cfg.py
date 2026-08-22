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

import isaaclab.utils.math as math_utils
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

        # v7 (2026-08-19, gate-fix -- claude-tg-base's diagnosis of the v6 failure,
        # verified against the code myself: the OLD `landing_active = elapsed >=
        # window_end` below was a pure TIMER, with no check that a real liftoff
        # ever happened. All four v6 positive landing-package terms (combined
        # weight 22) are gated on landing_active alone -- so they paid in full for
        # simply STANDING through the window and into the settle slice, no jump
        # required. A jump only ever RISKS that free payout (movement away from
        # the already-rewarded default stance costs elsewhere too), so the
        # dominant, cheapest strategy the policy actually found was standing
        # still and collecting the landing package uncontested -- exactly the
        # "щенячий прыжок, полу-ползание" the owner's live bench saw.
        #
        # Fix: `had_flight` is a per-env boolean LATCH, armed the first time all
        # four feet are simultaneously off the ground (same _feet_airborne
        # definition every other flight-gated term in this file already uses)
        # AND the min-foot-clearance has genuinely cleared FLIGHT_CLEARANCE_EPS
        # (2026-08-23 night: was MIN_FLIGHT_BASE_HEIGHT against root-Z, found to
        # be gameable by body pitch/rearing -- see FLIGHT_CLEARANCE_EPS's own
        # module-level comment for the full diagnosis) -- the extra
        # clearance check exists so a contact-sensor blip/stumble that clears the
        # ground for one physics step without real height can't arm the latch on
        # its own. Reset every cycle in _resample_command. landing_active now
        # ANDs this in: a cycle that never produced a real flight pays NOTHING
        # from the positive landing package, full stop.
        self.had_flight = torch.zeros(n, dtype=torch.bool, device=self.device)
        # Rising-edge flag for landing_active (this step is the FIRST step of the
        # landing slice for envs that actually flew) -- drives the new v7
        # episodic-style bonuses below, which must pay ONCE per successful jump
        # attempt, not every step of the whole landing slice (Atanassov's own
        # task_max_height/jumping pay once per episode at touchdown; our episode
        # holds many jump CYCLES, so this fires once per CYCLE instead -- see
        # jump_task_max_height's own docstring for the full adaptation note).
        self._prev_landing_active = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.landing_edge = torch.zeros(n, dtype=torch.bool, device=self.device)
        # Peak world-Z reached so far THIS CYCLE -- the episodic max_height
        # analog. Reset to the current (near-standing, since a fresh cycle always
        # starts in idle) height at each resample, same "reset to standing
        # baseline" idiom Atanassov's own self.max_height[env_ids] =
        # self.base_init_state[2] uses.
        self.cycle_peak_height = torch.zeros(n, device=self.device)
        # Cached contact-sensor resolution for the gate-fix's own airborne check
        # AND jump_change_of_contact below -- resolved once here via the same
        # SceneEntityCfg mechanism the reward manager itself uses (not the
        # module's free _feet_airborne helper, which expects a pre-resolved
        # sensor_cfg param a command term doesn't otherwise receive), same
        # pattern leg_lift_env_cfg.py's own v8 curriculum uses for command-term-
        # internal body/sensor lookups.
        self._feet_sensor_cfg = SceneEntityCfg("contact_forces", body_names=".*_calf")
        self._feet_sensor_cfg.resolve(self._env.scene)
        self._prev_contacts = torch.zeros(n, 4, dtype=torch.bool, device=self.device)
        self.contacts = torch.zeros(n, 4, dtype=torch.bool, device=self.device)
        self.contact_diff = torch.zeros(n, device=self.device)

        # v7d postmortem fix (2026-08-22, base's diagnosis, confirmed by code +
        # a direct yaw-drift-vs-lateral-drift correlation measurement --
        # TRAINING_STATE.md same date): jump_direction_velocity/
        # jump_flight_distance both used to recompute yaw from the LIVE
        # root_quat_w every single step and rotate world velocity into that
        # ever-changing frame -- "forward" for those terms meant "wherever the
        # torso currently points", not "wherever it pointed at launch". A
        # policy that yaws mid-flight can make world-frame lateral drift score
        # as honest "vel_along" for free, without any real progress toward the
        # originally commanded direction -- exactly the exploit that made
        # v7d's lateral drift grow 4x (0.06->0.23m) in lockstep with growing
        # yaw drift (9->22 deg) once jump_flight_distance's weight rose enough
        # to make the exploit worth it. Fix: capture yaw ONCE, the instant the
        # launch phase begins (phase>=CROUCH_PHASE_END), and hold it fixed for
        # the rest of that cycle -- same "resolve once per cycle, not every
        # step" idiom this class already uses for `direction` in
        # _resample_command. Both reward functions now read `term.launch_yaw`
        # instead of recomputing it.
        self.launch_yaw = torch.zeros(n, device=self.device)
        self._launch_yaw_captured = torch.zeros(n, dtype=torch.bool, device=self.device)

        # 2026-08-23 night redesign (base+train, root_pos_w-rearing-exploit fix
        # -- see FLIGHT_CLEARANCE_EPS's own comment above for the full
        # diagnosis): cached body indices for the min-foot-clearance
        # computation below, resolved once here (same idiom leg_lift_env_cfg.py's
        # own LegLiftCommand.__init__ uses for its _foot_ids). CALF bodies for
        # the foot-geom FK (body origin sits at the knee, not the foot -- see
        # FOOT_GEOM_LOCAL_OFFSET's own comment), HIP bodies for jump_vertical_
        # launch's new min-over-4-hips v_z (replaces root v_z, which rearing
        # could also cheaply fake).
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
        # Per-env min-across-4-feet ground clearance (flat terrain -> world Z
        # of the foot geom IS the clearance, no per-env baseline needed) --
        # computed once per step in _update_command, read by every reward
        # function that used to read root_pos_w for "how high is the jump".
        self.min_foot_clearance = torch.zeros(n, device=self.device)

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

        # v7 gate-fix state, reset for the fresh cycle these env_ids are starting.
        self.had_flight[env_ids] = False
        self._prev_landing_active[env_ids] = False
        self.landing_edge[env_ids] = False
        # v7 fix (2026-08-19, claude-tg-base's review, caught before launch):
        # clamped to STANDING_TARGET_HEIGHT, not the raw live root z. This
        # _resample_command call also fires on a fresh EPISODE reset (see
        # JumpPulseCommand's own class docstring / IsaacLab's own
        # CommandTerm.reset()->_resample() chain), and rough_env_cfg.py's
        # randomize_reset_base event spawns the robot at z up to +0.2m above
        # nominal AND at a fully random roll/pitch (+-pi) -- an uncapped read
        # here could seed cycle_peak_height near JUMP_TASK_MAX_HEIGHT_TARGET
        # (0.85) from spawn randomization alone, before the robot has taken a
        # single step, handing jump_task_max_height (weight 25) a large free
        # payout for doing nothing. Clamping the INITIAL value to the standing
        # target means every bit of credit toward the target must come from
        # genuine upward progress made DURING the cycle.
        # 2026-08-23 night redesign: cycle_peak_height now tracks min-foot-
        # CLEARANCE (see min_foot_clearance's own __init__ comment), not
        # root-Z -- a fresh cycle always starts in idle with all 4 feet
        # planted, so clearance is inherently ~0 regardless of spawn-
        # randomization noise (root_pos_w/roll/pitch can spawn up to +-pi and
        # +0.2m under rough_env_cfg.py's randomize_reset_base event -- the OLD
        # STANDING_TARGET_HEIGHT clamp existed specifically to neutralize
        # that for the root-Z version of this state; clearance doesn't need
        # an equivalent clamp because standing-on-the-ground clearance is
        # already ~0 by construction, not vulnerable to the same exploit).
        self.cycle_peak_height[env_ids] = 0.0
        # v7d fix: fresh cycle, launch_yaw not captured yet this time around.
        self._launch_yaw_captured[env_ids] = False

    def _update_command(self):
        elapsed = self.cycle_duration - self.time_left
        window_start = self.idle_duration
        window_end = self.idle_duration + self.cfg.window_duration
        self.window_active = (elapsed >= window_start) & (elapsed < window_end)

        # v7 gate-fix: arm had_flight the first time all four feet are genuinely
        # airborne (contact-sensor fact, not a visual guess) AND the min-foot-
        # clearance has genuinely cleared FLIGHT_CLEARANCE_EPS -- see this
        # attribute's own __init__ comment for the full diagnosis this fixes,
        # and FLIGHT_CLEARANCE_EPS's own module-level comment for why root-Z
        # was replaced here (2026-08-23 night, rearing exploit fix).
        contact_sensor = self._env.scene.sensors[self._feet_sensor_cfg.name]
        in_contact = contact_sensor.data.current_contact_time[:, self._feet_sensor_cfg.body_ids] > 0.0
        self.contacts = in_contact
        # v7: per-step contact-state churn, for jump_change_of_contact below --
        # Atanassov's own change_of_contact, computed once here (not inside the
        # reward function) since it needs the PREVIOUS step's contact state,
        # which only the command term persists across steps.
        self.contact_diff = torch.sum(torch.abs(self.contacts.float() - self._prev_contacts.float()), dim=1)
        self._prev_contacts = self.contacts.clone()
        all_airborne = ~in_contact.any(dim=1)

        # min-across-4-feet ground clearance (2026-08-23 night redesign) --
        # FK from each calf BODY's own pose (frame origin at the knee) plus
        # the fixed local offset to the actual foot geom (see FOOT_GEOM_
        # LOCAL_OFFSET's own comment). Flat terrain -> world Z of the foot
        # geom IS the clearance directly, no per-env ground-height baseline
        # needed (same "flat plane -> plain world-Z" convention this file's
        # own jump_idle_height/jump_crouch already use).
        calf_pos_w = self.robot.data.body_pos_w[:, self._calf_body_ids, :]  # (n,4,3)
        calf_quat_w = self.robot.data.body_quat_w[:, self._calf_body_ids, :]  # (n,4,4)
        n_envs = calf_pos_w.shape[0]
        local_offset = self._foot_local_offset.expand(n_envs, 4, 3)
        world_offset = math_utils.quat_apply(
            calf_quat_w.reshape(-1, 4), local_offset.reshape(-1, 3)
        ).reshape(n_envs, 4, 3)
        foot_pos_w = calf_pos_w + world_offset
        self.min_foot_clearance = foot_pos_w[:, :, 2].min(dim=1).values

        self.had_flight = self.had_flight | (
            self.window_active & all_airborne & (self.min_foot_clearance > FLIGHT_CLEARANCE_EPS)
        )
        # 2026-08-23 night, base's review (BUG-1): must be gated to window_active,
        # NOT unconditional every step. rough_env_cfg.py's randomize_reset_base
        # event spawns the robot up to +0.2m above the ground at EPISODE reset --
        # during the idle slice right after that (window_active is False there,
        # nowhere near a real jump attempt), the robot is still settling/falling
        # from spawn, briefly clearing all 4 feet with real clearance. Unconditional
        # tracking let that spawn-fall get recorded as this cycle's "peak", handing
        # jump_task_max_height a large free payout for a later 4cm micro-hop that
        # had nothing to do with the spawn event -- and worse, a policy terminating
        # itself (illegal_contact) could farm repeated spawn-falls for free credit.
        # window_active alone is sufficient (excludes idle entirely, where the old
        # code's own spawn-fall bug lived) -- a real flight peak only ever happens
        # inside the window by construction.
        self.cycle_peak_height = torch.where(
            self.window_active,
            torch.max(self.cycle_peak_height, self.min_foot_clearance),
            self.cycle_peak_height,
        )

        landing_timer = elapsed >= window_end
        self.landing_active = landing_timer & self.had_flight
        self.landing_edge = self.landing_active & ~self._prev_landing_active
        self._prev_landing_active = self.landing_active.clone()

        self.phase = torch.where(
            self.window_active,
            ((elapsed - window_start) / self.cfg.window_duration).clamp(0.0, 1.0),
            torch.zeros_like(elapsed),
        )
        active = self.window_active.unsqueeze(-1).float()
        self._command[:, 0:2] = self.direction * active
        self._command[:, 2] = self.phase

        # v7d fix: capture yaw ONCE, the first step this cycle where the
        # launch phase has begun -- see launch_yaw's own __init__ comment.
        entering_launch = self.window_active & (self.phase >= CROUCH_PHASE_END) & ~self._launch_yaw_captured
        if torch.any(entering_launch):
            q = self.robot.data.root_quat_w
            w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
            yaw_now = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
            self.launch_yaw = torch.where(entering_launch, yaw_now, self.launch_yaw)
            self._launch_yaw_captured = self._launch_yaw_captured | entering_launch


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

# Named 2026-08-19 (v7 gate-fix) -- was already a bare 0.55 literal passed to
# jump_flight/jump_flight_distance's own min_base_height param below (their
# anti-squat-pogo gate, 2026-08-05). Named now because JumpPulseCommand's own
# had_flight latch (see its __init__ comment) reuses the EXACT same value for
# the exact same purpose ("did a real jump happen, not just ground noise") --
# a shared named constant keeps the two uses from silently drifting apart if
# either is ever retuned.
#
# 0.55 -> 0.54 (2026-08-21, v7 А overnight monitoring, resumed from v5's
# model_44998.pt): 6 checkpoint-spaced peak-height measurements
# (scratchpad/check_jump_peak.py, precise MuJoCo bench readout, not eyeballed)
# over ~1h of training at it49400-55000 came back 0.5458/0.5447/0.5440/0.5430/
# 0.5430 -- a confirmed, stable ceiling just under the gate, not noise (reward
# and vertical_launch/direction_velocity stayed healthy/growing the whole
# time, only the peak-height metric specifically stalled). Because had_flight
# (and everything downstream: jump_flight, jump_task_max_height,
# jump_task_jumping_bonus, all the jump_landing_* terms) NEVER latches until
# a cycle clears this bar even once, the policy had zero gradient toward the
# real 0.85m target (JUMP_TASK_MAX_HEIGHT_TARGET) despite being ~1.3cm above
# standing height (0.53) already -- a classic RL bootstrap dead zone, not a
# capability ceiling.
#
# CORRECTION, same session, ~30min later: 0.54 didn't unblock had_flight even
# though the resumed run's own peak-height readout (0.5432m) technically
# cleared it -- check_jump_peak.py measured the ABSOLUTE max base_z over the
# whole window regardless of foot contact, but had_flight requires base_z>
# threshold AND all_airborne SIMULTANEOUSLY (see _update_command below). Wrote
# scratchpad/check_jump_peak3.py to log base_z specifically during genuinely-
# all-airborne instants: on the same it56000 checkpoint, that narrower true
# max was only 0.5397m -- the ballistic peak happens slightly AFTER the rear
# feet already start re-contacting for landing, so "highest point reached" and
# "highest point while still fully airborne" are different numbers, and only
# the second one is what this gate actually checks. 0.54 -> 0.535 (this
# comment's edit): ~3mm below the measured true airborne-max (0.5397) as a
# safety margin, not 1cm above the WRONG number 0.5430 was read as. Still
# comfortably above standing height (0.53, ~5mm) and the squat-pogo exploit
# floor (~0.2m) -- same anti-reopening reasoning as the original 0.54 pick,
# just recalibrated to the metric the code actually gates on.
#
# Resumed training from the last checkpoint before EITHER edit (model_55000.pt
# from run 2026-08-21_03-08-57), not restarted from scratch either time.
#
# REMOVED FROM THE REWARD LOOP ENTIRELY (2026-08-23 night, base+train, direct
# owner order after a live bench test of it67000 found "щенячий прыжок" --
# see train_research/TRAINING_STATE.md/TRAIN_RESEARCH.md same date for the
# full diagnosis). This constant compared against root_pos_w[:,2] -- but
# MIN_FLIGHT_BASE_HEIGHT (0.535) sat only 5mm above STANDING_TARGET_HEIGHT
# (0.53), AND root-Z is trivially raised by pitching the body nose-up
# (rearing) without any real vertical liftoff: at ~1.1m body length, a
# 25-30deg pitch alone clears this bar. Confirmed by direct frame inspection
# (visual_policy_check.py, it67000 frame_0022): base_z=0.595m ("PASS with
# margin" by this now-dead threshold) while BOTH rear feet were still in
# ground contact. Every consumer of this constant (had_flight,
# jump_flight/jump_flight_distance's own `high` gate, jump_task_jumping_
# bonus's height_threshold) now reads FLIGHT_CLEARANCE_EPS against
# min-foot-clearance instead -- a quantity rearing/pitching cannot fake (the
# LOWEST foot has to be genuinely off the ground, regardless of body
# attitude). See JumpPulseCommand._update_command's own min_foot_clearance
# computation.
#
# Small positive floor (not 0.0) purely to reject contact-sensor jitter at
# true ground level -- NOT a height target itself (that's JUMP_TASK_MAX_
# HEIGHT_TARGET below). Same order of magnitude as the noise floor other
# contact-based terms in this file already tolerate.
FLIGHT_CLEARANCE_EPS = 0.03  # m

# Named 2026-08-19 (v7, cycle_peak_height's own init-clamp fix) -- was already
# a bare 0.53 literal at jump_landing_height_stance/jump_idle_height's own
# target_height params below (rough's own standing target). Named here
# because the clamp fix needs the exact same number for the exact same
# reason ("this is what a legitimate standing pose's height is"); the two
# pre-existing literal call sites are left as-is (unrelated to this fix,
# not touched to keep the diff scoped).
STANDING_TARGET_HEIGHT = 0.53

# v7 (2026-08-19, Atanassov-style episodic bonuses, see jump_task_max_height/
# jump_task_jumping_bonus's own docstrings). Absolute world-Z peak-height
# target -- NOT copied from Atanassov's own 0.9m (their Go1 stands at 0.32m,
# so 0.9m is ~0.58m of clearance, ~1.8x their own standing height; scaling
# that SAME ratio to B2's 74.5kg/0.53m stance would demand ~0.95m of
# clearance, physically implausible for a robot this heavy -- torque/mass
# doesn't scale the way a small-robot ratio would suggest). Derived instead
# from B2's own leg geometry, same FK discipline as leg_lift_env_cfg.py's own
# THIGH_FOLD_TARGET reasoning: b2 URDF thigh-to-calf link is 0.35m, so a
# fully explosive extension could plausibly clear somewhere near one
# thigh-length above stance in a genuine dynamic launch (more than the
# 0.30m leg_lift v8 used for a STATIC fold, since a jump adds real upward
# velocity on top of pure geometry) -- 0.53 (jump_landing_height_stance's own
# standing target) + 0.35 = 0.88, rounded to 0.85 to leave the same kind of
# headroom-from-an-unverified-ceiling margin every other first-guess target
# in this codebase leaves. FIRST GUESS, not vendor-sourced (Unitree's own B2
# marketing gives only a >1.6m horizontal LENGTH figure, no vertical number --
# see TRAIN_RESEARCH.md's [WEB] entry) -- expect postmortem-driven revision.
# 0.85 (root-Z target) -> 0.20 (2026-08-23 night, redefined to min-foot-
# clearance target, same base+train redesign as FLIGHT_CLEARANCE_EPS above):
# `term.cycle_peak_height` now tracks peak min-foot-clearance, not peak
# root-Z (see JumpPulseCommand._update_command) -- this target must be
# re-derived in the SAME units. 0.20m is the FIRST RUNG of the owner's own
# ladder toward his real target (foot clearance 0.50-0.60m, corpus apex
# over ~1m -- his explicit number, 2026-08-23: "высота нижней части ног
# должна быть не менее 500-600мм"). Ladder: 10cm (first honest liftoff,
# the bench's own gate first rung) -> 20 -> 35 -> 50-60cm, moved by hand
# between runs (no curriculum mechanics -- fewer moving parts, matches this
# file's own existing convention of hand-tuned weight steps, not automatic
# curricula). Retarget this constant to the NEXT rung once a run clears the
# current one -- do not leave stale between rungs.
JUMP_TASK_MAX_HEIGHT_TARGET = 0.20
# 0.08 -> 0.02 (same redesign): must stay a LIVE gradient at clearance=0 (a
# fresh/early checkpoint's honest starting point) -- verified numerically
# before launch: exp(-(0-0.20)**2/0.02) = exp(-2.0) = 0.135, comfortably
# above the "not vanishing" floor (>=0.02) base's review required. Sharper
# than the old 0.08 because the new target (0.20m) is a much smaller
# absolute scale than the old one (0.85m) -- keeping the OLD sigma here
# would have made the kernel too flat to discriminate progress within the
# ladder's own 10-60cm range.
JUMP_TASK_MAX_HEIGHT_SIGMA = 0.02

# CALF_TUCK_TARGET / TUCK_SIGMA / jump_airborne_leg_tuck -- ADDED then
# REMOVED same night (2026-08-23), after the owner shared official Unitree
# B2 jump reference footage (train_research/ROBOTS/B2/JUMP_REFERENCE.md):
# the vendor video shows legs progressively EXTENDING and reaching forward
# through flight, not folding/tucking toward the body -- the FK-calibrated
# fold-toward-body pose this constant targeted was the wrong shape entirely.
# Rather than invent a SECOND choreographed pose target (a "reach forward"
# anchor) to match the corrected reference, base's design reuses the
# ALREADY-CALIBRATED jump_landing_pose anchor for the descending half of
# flight instead (see that function's own 2026-08-23 comment) -- simpler,
# non-gameable (gated on ballistic v_z sign, not a policy-chosen pose), and
# avoids repeating the same prescribe-a-pose mistake with different numbers.

# FK offset from each calf BODY's own frame origin (which sits at the KNEE,
# not the foot -- b2.xml has no separate foot bodies) to the actual foot
# COLLISION geom, confirmed by direct geom enumeration (2026-08-23, base's
# review caught this: using the calf body origin directly for any
# clearance-based metric would itself be a manipulable anchor of the same
# class this whole redesign exists to eliminate -- straightening a planted
# leg raises the knee several cm with zero real liftoff). All 4 legs share
# this offset to a very close tolerance (the small per-leg mesh asymmetry
# is sub-millimeter, verified negligible). Same technique already used in
# scripts/check_jump_liftoff_quality.py's own _foot_geom_ids, just applied
# here via a fixed local-frame constant (a torch tensor + quat_apply) since
# training-time reward code operates on IsaacLab's body-level state, not a
# raw MuJoCo model to enumerate geoms from directly.
#
# -0.35 -> -0.382 (2026-08-23, base's review caught a second-order version
# of the SAME manipulable-anchor class): the foot collision geom is a
# SPHERE, radius 0.032m (confirmed by direct MuJoCo model inspection,
# geom_size[0]=0.032) -- -0.35 alone lands on the sphere's CENTER, not
# where it touches the ground. At standing pose this read ~0.059m instead
# of ~0m -- FLIGHT_CLEARANCE_EPS (0.03) and JUMP_TASK_MAX_HEIGHT_TARGET's
# own zero-point would have been silently offset by the sphere radius,
# reading "genuine liftoff" a full 3cm before the foot actually left the
# ground. -0.35 - 0.032 = -0.382 reaches the sphere's BOTTOM (the true
# ground-contact point) instead -- verified: standing clearance now reads
# 0.027m (residual is normal contact-solver penetration/settle, not a
# calibration error), no longer conflated with the sphere's own radius.
FOOT_GEOM_LOCAL_OFFSET = (0.0, 0.0, -0.382)


def _feet_airborne(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """True per env when ALL feet are out of contact (the flight phase)."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    return ~in_contact.any(dim=1)


def jump_flight(env, command_name: str, sensor_cfg: SceneEntityCfg, min_clearance: float) -> torch.Tensor:
    """Flat per-step bonus for being fully airborne INSIDE the jump window -- makes
    the very first accidental hop immediately rewarding, before any distance is
    covered (bootstrap term).

    min_clearance gate (2026-08-05, root-Z; RETARGETED 2026-08-23 night to
    min-foot-clearance, base+train redesign): the first from-scratch run
    reward-hacked the ungated version -- it sat compressed at ~0.2m through
    idle and pogo-trembled in the windows (bench-measured: crouch 0.20,
    "peak" 0.39, zero displacement, never reaches standing height at all),
    collecting airborne ticks for micro-hops. The ORIGINAL fix priced this
    against root_pos_w[:,2] > MIN_FLIGHT_BASE_HEIGHT -- which stopped
    micro-hops but turned out to itself be gameable a different way: body
    PITCH (rearing) raises root-Z with zero real liftoff (see
    FLIGHT_CLEARANCE_EPS's own comment for the full diagnosis). Now gates on
    min-foot-clearance instead -- the lowest foot must be genuinely off the
    ground, a quantity rearing/pitching cannot fake."""
    term = env.command_manager.get_term(command_name)
    high = (term.min_foot_clearance > min_clearance).float()
    post_crouch = (term.window_active & (term.phase >= CROUCH_PHASE_END)).float()
    return _feet_airborne(env, sensor_cfg).float() * high * post_crouch


def jump_flight_distance(
    env, command_name: str, sensor_cfg: SceneEntityCfg, max_vel: float, min_clearance: float
) -> torch.Tensor:
    """Velocity along the commanded direction WHILE AIRBORNE inside the window --
    integrates to "distance covered in flight", which is the actual definition of a
    jump (a run covers distance grounded and earns nothing here). Clipped at max_vel
    so ballistic distance, not raw launch violence, is what pays."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    # Same min_clearance gate as jump_flight -- see its docstring (anti-squat-
    # pogo AND anti-rearing, 2026-08-23 retarget from root-Z to foot clearance).
    high = (term.min_foot_clearance > min_clearance).float()
    # v7d fix (2026-08-22): rotate into the YAW FIXED AT LAUNCH (term.launch_yaw),
    # not the live root_quat_w recomputed every step -- see launch_yaw's own
    # __init__ comment. The old "cheap 2D rotation via the CURRENT yaw" let a
    # policy that yaws mid-flight collect world-frame lateral drift as free
    # "vel_along" credit -- confirmed by a direct yaw-drift-vs-lateral-drift
    # correlation measurement (TRAINING_STATE.md 2026-08-22, v7d postmortem).
    cos_yaw, sin_yaw = torch.cos(term.launch_yaw), torch.sin(term.launch_yaw)
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
    """Dense bootstrap: upward LEG velocity inside the jump window pays IMMEDIATELY,
    grounded or not. Added 2026-08-04 after the first v3 run proved the flight terms
    alone can't bootstrap: both are gated on ALL FOUR feet already being airborne, an
    event a standing policy never produces by exploration noise, so 6000 iterations
    passed with jump_flight pinned at exactly zero while the policy settled into
    "stand through the window" (~200 reward from stand-still/regularizers) and PPO's
    noise_std climbed back up hunting a gradient that wasn't there. Any upward push
    now climbs toward the real jump: harder push -> higher v_z -> eventually flight,
    where the flight-gated distance term takes over. Clamped so a real launch
    (v_z ~2 m/s) dominates the crumbs a trot's own bounce could collect.

    2026-08-23 night (base+train redesign, rearing-exploit fix): root
    `asset.data.root_lin_vel_w[:,2]` REPLACED with `min-over-4-HIP-bodies
    v_z` -- root v_z is exactly as fakeable by pitching (rearing) as
    root_pos_w itself was (a body pitching nose-up has a rising root
    velocity too, without any leg genuinely pushing). During rearing/
    kneeling the REAR hips barely rise (they stay near the ground) --
    min-over-4-hips collapses to ~0 in that case, killing the credit at
    its SOURCE. A genuine level push raises all 4 hips together and pays
    in full -- this is the "толкайся ЗАДНИМИ" lever the owner asked for,
    expressed as a reward gradient rather than a prescribed pose (the
    launch phase itself gets no choreographed pose target anywhere in
    this file's 2026-08-23 redesign -- see jump_landing_pose's own comment
    for why the descending-flight pose anchor exists instead)."""
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
    # (direction 8 vs vertical 4). Cap kept at the SAME 3.0 for the new min-hip-v_z
    # quantity -- same physical units, same "a real push dominates trot bounce" logic.
    hip_v_z = asset.data.body_lin_vel_w[:, term._hip_body_ids, 2]  # (n,4)
    min_hip_v_z = hip_v_z.min(dim=1).values
    return min_hip_v_z.clamp(0.0, 3.0) * in_launch.float()


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
    """Penalize roll/pitch (body tilting away from level) during flight and
    landing (added 2026-08-12, bench verdict on 34999: the backward jump
    lifts the rear high into the air -- "попу поднимает сильно" -- and
    barely lands on balance). The inherited `upward` reward (3.0) prices this
    too weakly against launch terms at 8: a hind-dominant push can buy a big
    pitch-up cheaply. Squared world-frame roll+pitch rate is NOT used --
    attitude itself is the problem, so price the gravity-projected tilt
    directly (same quantity flat-orientation terms use everywhere else).

    2026-08-23 night (base+train redesign): gate window NARROWED from "all
    post-crouch" (launch push + flight + landing) to airborne-or-landing
    ONLY -- excludes the launch push itself. Term-attribution on it67000
    (TRAINING_STATE.md same date) found this term WAS already firing during
    rearing (pitch -18.9 -> -35.75deg, penalty growing -0.22 -> -0.68) but
    got outbid by the height/velocity credit that pitch could fake -- not a
    gating problem, a magnitude-vs-jump_vertical_launch problem. Now that
    jump_vertical_launch itself no longer credits rearing (min-over-4-hips
    v_z collapses near zero when the rear hips stay low), a real rear-
    dominant push's OWN transient pitch during the explosive launch instant
    is safe to allow -- a real jump physically passes through some
    momentary tilt during push-off; only flight/landing attitude matters
    for a clean ballistic trajectory and a controlled touchdown."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene["robot"]
    # projected_gravity_b xy components are 0 when the body is level.
    g_xy = asset.data.projected_gravity_b[:, 0:2]
    tilt = torch.sum(torch.square(g_xy), dim=1)
    all_airborne = ~term.contacts.any(dim=1)
    # 2026-08-23 night, base's review (BUG-2): `landing_active` is BY
    # DEFINITION `elapsed >= window_end & had_flight` while `window_active`
    # requires `elapsed < window_end` -- the two are mutually exclusive, so
    # the ORIGINAL `window_active & (all_airborne | landing_active)` made
    # `window_active & landing_active` always False, silently killing the
    # landing half of this term entirely (attitude was NEVER actually priced
    # on landing, despite the docstring's own claim). Parenthesization fixed:
    # window_active gates ONLY the airborne branch (a real flight moment is
    # always inside the window by construction); landing_active already
    # carries its own sufficient gating (implies past window_end AND a real
    # flight happened), doesn't need window_active additionally.
    in_free = (term.window_active & all_airborne) | term.landing_active
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
    # v7d fix (2026-08-22): use the YAW FIXED AT LAUNCH (term.launch_yaw), not
    # the live root_quat_w recomputed every step -- see launch_yaw's own
    # __init__ comment for the yaw-drift exploit this closes. Captured on the
    # exact same step `in_launch` below first goes true, so it's never stale.
    cos_yaw, sin_yaw = torch.cos(term.launch_yaw), torch.sin(term.launch_yaw)
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
    expect revision same as every other sigma in this codebase's history.

    2026-08-23 night (base's design, vendor-reference-driven -- see
    train_research/ROBOTS/B2/JUMP_REFERENCE.md, official Unitree B2 jump
    footage the owner shared): gate window WIDENED from `landing_active`
    alone to `landing_active | (airborne & descending)`. The reference
    video shows the legs progressively EXTENDING and reaching forward
    through flight, converging on something close to this SAME default-
    stance pose right before touchdown (frame 4: "ещё сильнее выпрямлены,
    подала их вперёд" -- reaching for the landing). An earlier design
    attempted a separate FK-calibrated "tuck" pose anchor for the rising
    half of flight (jump_airborne_leg_tuck, since REMOVED) -- the vendor
    footage showed that was the wrong shape entirely (extension/reach, not
    a fold), and inventing a SECOND choreographed "reach" pose target would
    have repeated the same prescribe-a-pose mistake with a different pose,
    not fixed it. Reusing this ALREADY-CALIBRATED anchor for the descent
    half of flight is simpler and non-gameable: `descending` (root v_z < 0)
    is decided by gravity once airborne, not something the policy can fake
    -- same "outcome-based, not choreographed" principle as the vertical_
    launch fix above. The RISING half of flight gets no pose anchor at all
    (legs are already extended from the push-off itself, physically nothing
    to anchor there per the reference -- see jump_env_cfg.py's design
    notes for the full reasoning)."""
    term = env.command_manager.get_term(command_name)
    asset = env.scene[asset_cfg.name if asset_cfg is not None else "robot"]
    joint_names = asset.joint_names
    leg_ids = [i for i, n in enumerate(joint_names) if n.endswith(("_hip_joint", "_thigh_joint", "_calf_joint"))]
    err = torch.sum(torch.square(asset.data.joint_pos[:, leg_ids] - asset.data.default_joint_pos[:, leg_ids]), dim=1)
    all_airborne = ~term.contacts.any(dim=1)
    descending = asset.data.root_lin_vel_w[:, 2] < 0.0
    gate = term.landing_active | (all_airborne & descending)
    return torch.exp(-err / JUMP_LANDING_POSE_SIGMA) * gate.float()


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


# v7 (2026-08-19): Atanassov et al.'s episodic (paid-once-at-touchdown) reward
# style, ported from their own legged_gym source (github.com/vassil-atn/
# Curriculum-Quadruped-Jumping-DRL, legged_robot.py::_reward_task_max_height/
# _reward_jumping -- checked directly, not from a paraphrase, see
# TRAIN_RESEARCH.md). Structural adaptation, documented up front: their
# episode ends at (or shortly after) the one jump it contains, so "once per
# episode" and "once at touchdown" are the same event; OUR episode holds MANY
# jump cycles back to back (idle->window->landing->resample, repeated for the
# whole episode length), so these fire once per CYCLE instead, on
# `term.landing_edge` (the first step landing_active turns true for a cycle
# that actually had a flight -- see JumpPulseCommand.__init__'s own comment).
# Weights are NOT copied from their 5000/200 -- see the registration site
# below for why those numbers don't transfer to this file's own scale.
def jump_task_max_height(env, command_name: str) -> torch.Tensor:
    """Positive, paid ONCE per cycle at the landing_edge: exp-tracking of this
    cycle's peak world-Z (term.cycle_peak_height) toward
    JUMP_TASK_MAX_HEIGHT_TARGET. Direct port of Atanassov's own
    `_reward_task_max_height` (`exp(-(max_height-0.9)**2 / sigma)`, gated to
    `has_jumped` envs at episode end) -- same shape, per-cycle instead of
    per-episode, target rederived for B2 (see JUMP_TASK_MAX_HEIGHT_TARGET's
    own comment). Naturally zero for a cycle that never had a real flight:
    landing_edge itself can only ever be True once had_flight has latched
    (landing_active = landing_timer & had_flight), so no separate had_flight
    check is needed here -- the v6 failure mode (paid for standing still) is
    structurally impossible for an edge-triggered term."""
    term = env.command_manager.get_term(command_name)
    err = torch.square(term.cycle_peak_height - JUMP_TASK_MAX_HEIGHT_TARGET)
    return torch.exp(-err / JUMP_TASK_MAX_HEIGHT_SIGMA) * term.landing_edge.float()


def jump_task_jumping_bonus(env, command_name: str, height_threshold: float) -> torch.Tensor:
    """Positive, paid ONCE per cycle at landing_edge: flat bonus if this
    cycle's peak height cleared `height_threshold`. Direct port of
    Atanassov's own `_reward_jumping` (binary, `max_height>0.50` at episode
    end) -- coarse complement to jump_task_max_height's own sharp exp-kernel,
    same "cheap bootstrap alongside a precise shaping term" relationship
    jump_flight(8)/jump_flight_distance(8) already have in this file.
    height_threshold reuses FLIGHT_CLEARANCE_EPS (2026-08-23 night: was
    MIN_FLIGHT_BASE_HEIGHT, retargeted alongside cycle_peak_height's own
    switch from root-Z to min-foot-clearance -- the same "did a real jump
    happen" bar the gate-fix itself uses) rather than a new number."""
    term = env.command_manager.get_term(command_name)
    return (term.cycle_peak_height > height_threshold).float() * term.landing_edge.float()


def jump_change_of_contact(env, command_name: str) -> torch.Tensor:
    """Continuous, every step: rewards the four feet's contact STATE staying
    UNCHANGED between consecutive steps (term.contact_diff, computed inside
    JumpPulseCommand._update_command since it needs the previous step's
    contacts). Direct port of Atanassov's own `_reward_change_of_contact`
    (`exp(-diff**2/4)`) -- HIGH when contact is stable (diff=0), LOW when it's
    churning, same as here.
    v7 CORRECTION (2026-08-19, claude-tg-base's independent review caught
    this before launch): this function was always right, but the
    REGISTRATION below originally applied a NEGATIVE weight on the mistaken
    belief that Atanassov register theirs as a penalty -- checked their
    go1_config.py again: `change_of_contact = 10.0`, POSITIVE. A negative
    weight on an exp-kernel that's HIGH at zero churn does not "penalize
    churn" -- it penalizes STABILITY and rewards churn (idle standing still,
    diff=0 nearly every step, would have paid -3.0/step; a foot chattering
    wildly would pay near 0, i.e. relatively BETTER). Fixed to a positive
    weight, matching Atanassov's own sign. Not phase-gated -- active through
    idle/crouch/launch/flight/landing alike, matching their own always-on
    term (their 0.5x post-landing scale-down is NOT ported: "has_jumped" in
    their code means "anywhere past this episode's one jump", which doesn't
    map onto our repeating-cycle structure without extra state; flagged here
    as a known simplification, not an oversight)."""
    term = env.command_manager.get_term(command_name)
    return torch.exp(-torch.square(term.contact_diff) / 4.0)


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
                # 2026-08-23 night: retargeted root-Z -> min-foot-clearance
                # (see FLIGHT_CLEARANCE_EPS's own comment for why).
                "min_clearance": FLIGHT_CLEARANCE_EPS,
            },
        )
        # 8.0 -> 11.0 (2026-08-22, owner's direct decision after the vertical_launch
        # 10->14 fix: peak height gained margin (0.5435->0.5493m, 6/6 dense sweep,
        # see TRAINING_STATE.md ~14:30) but forward distance dropped -45% (0.225m
        # ->0.124m) as a measured, consistent side-effect -- raising vertical
        # impulse without touching horizontal terms shifted the economy toward
        # height at distance's expense. Single-variable fix, same discipline as
        # the vertical_launch change: this term is gated on min_base_height
        # (MIN_FLIGHT_BASE_HEIGHT, the same bar just confirmed reliably cleared)
        # so pushing it can't undermine the height fix -- it only pays once the
        # jump has already cleared that bar, rewarding MORE horizontal velocity
        # during an already-qualifying flight, not a competing objective.
        # +37.5% (not a multiplier), same "one deliberate step" precedent as the
        # vertical_launch +40%. Resume from jump_v7c_final_65499 (the checkpoint
        # this regression was measured on), --no_resume_optimizer (new weight
        # scale), short budget to see the trend before committing more.
        #
        # 11.0 -> 8.0 REVERTED (2026-08-22 night, v7d exploit-fix verification
        # run, base+train, see train_research/TRAINING_STATE.md same date):
        # 5-checkpoint dense sweep on the it65499->67498 run (SAME resume
        # source, WITH the launch_yaw exploit fix from commit 9e90304 applied)
        # showed forward distance never recovered at all across the whole
        # 2000-iteration budget (0.122/0.093/0.092/0.078/0.103m -- still far
        # below v7c's own 0.225m baseline) -- the weight bump itself doesn't
        # achieve its goal, exploit or not. Reverting to v7c's own value;
        # recovering distance needs a different approach (not just this one
        # weight), left as an open problem for a future session, not tonight.
        # The exploit fix itself (launch_yaw) stays -- structurally correct
        # regardless of this experiment's outcome. v7c + the fix is now the
        # stable baseline going forward.
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
                # 2026-08-23 night: retargeted root-Z -> min-foot-clearance.
                "min_clearance": FLIGHT_CLEARANCE_EPS,
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
        # jump_airborne_leg_stillness (2026-08-09, velocity PENALTY) and
        # jump_airborne_front_leg_pose (2026-08-14, FRONT-leg-only pose
        # anchor) both DELETED 2026-08-23 night (base+train redesign,
        # rearing-exploit fix). An intermediate design (jump_airborne_leg_
        # tuck, a fold-toward-body pose anchor) was tried and ALSO removed
        # the same night after the owner shared official Unitree B2 jump
        # reference footage (train_research/ROBOTS/B2/JUMP_REFERENCE.md)
        # showing the real jump EXTENDS the legs through flight, not folds
        # them -- see jump_landing_pose's own 2026-08-23 comment for the
        # design that replaced it (widened that already-calibrated anchor's
        # gate to cover the descending half of flight too, instead of
        # inventing a new "reach forward" pose target). The rising half of
        # flight gets NO pose anchor at all: per the reference, legs are
        # already extended right out of the push-off, nothing to anchor.
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
        # 10.0 -> 14.0 (2026-08-22, post-leg_lift methodology transfer): dense
        # checkpoint sweep on the v7a3 tail (it55000-64999, step ~100-1000)
        # found peak airborne base_z hovering RIGHT at MIN_FLIGHT_BASE_HEIGHT
        # (0.535) with checkpoint-to-checkpoint noise flipping PASS/FAIL every
        # ~200-300 iterations -- confirmed real (bit-identical repeats on a
        # deterministic bench) and confirmed NOT a training-dynamics glitch
        # (Loss/entropy, Loss/learning_rate, Loss/value_function, and every
        # Episode_Reward/* scalar are flat/noisy across the whole tail with no
        # signature). Actuator torque checked directly against the bench's
        # own model.actuator_ctrlrange (200/200/300 Nm hip/thigh/calf) on
        # it63500's launch phase: peak 83.3% of cap (RR_thigh), 0.0% of steps
        # within 1% of saturation on any of the 12 joints -- the ceiling is
        # reward-shaped, not physical. Measured peak v_z at launch directly:
        # 1.30 m/s, only 43% of this term's own clamp(0,3.0) -- vertical_launch
        # itself is far from saturated either. Ballistics (Δh = (v2²-v1²)/2g)
        # says reaching v_z~1.51 (+16%) buys ~3cm of extra peak height, a
        # comfortable margin over the 0.535 threshold instead of sitting
        # exactly on it. jump_task_max_height (weight 25, already the file's
        # largest single term) was considered instead but rejected: it's paid
        # ONCE per cycle at landing_edge (sparse) vs vertical_launch's every-
        # launch-step density, and its own target (JUMP_TASK_MAX_HEIGHT_TARGET
        # =0.85) is explicitly marked "FIRST GUESS...expect postmortem-driven
        # revision" in its own comment -- changing two unverified numbers at
        # once (an untested target AND its weight) isn't an isolated
        # experiment. +40% (not a multiplier) per the leg_lift lesson that a
        # single well-aimed step, not a large jump, produced a clean
        # qualitative fix rather than an uncontrolled one. Resume from
        # it63500 (best-margin checkpoint of the noisy tail, NOT claimed
        # "safe" -- see TRAINING_STATE.md 2026-08-22) with --no_resume_optimizer
        # (this file's own weight change invalidates the old Adam statistics
        # tuned for weight=10.0's gradient scale). Budget ~1500-2000 it to see
        # trend before committing more. Gate: dense checkpoint sweep (not one
        # point) AND the same 4 landing metrics that are currently clean
        # (impact/settle/bounce/drift) -- a height win that quietly breaks
        # landing is the same "one fix, new symptom" pattern leg_lift's
        # jerkiness co-symptom already taught this lab once.
        self.rewards.jump_vertical_launch = RewTerm(
            func=jump_vertical_launch,
            weight=14.0,
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

        # v7 (2026-08-19): Atanassov-style episodic bonuses (see the three
        # functions' own docstrings). Weights independently recalibrated for
        # THIS file's own scale, NOT copied from Atanassov's 5000/200/10 --
        # their numbers are meaningful within their own reward SUM (which they
        # additionally post-process via only_positive_rewards_ji22_style, a
        # global positive/negative split this codebase's manager-based
        # RewardManager has no hook for -- see TRAIN_RESEARCH.md for why that
        # piece was deliberately NOT ported rather than faked). Everything
        # else in this file lives in the 0.5-10 range; a literal 5000 would
        # swamp every other term into irrelevance. jump_task_max_height set
        # ABOVE the existing landing package (22 combined) and the launch
        # anchor (vertical_launch 10) -- it is meant to be the single largest
        # incentive in the file, matching Atanassov's own relative emphasis
        # ("их главная тяга") without adopting their absolute scale.
        self.rewards.jump_task_max_height = RewTerm(
            func=jump_task_max_height,
            weight=25.0,
            params={"command_name": "base_velocity"},
        )
        self.rewards.jump_task_jumping_bonus = RewTerm(
            func=jump_task_jumping_bonus,
            weight=6.0,
            # 2026-08-23 night: retargeted root-Z -> min-foot-clearance (this
            # reads term.cycle_peak_height, which now tracks peak clearance --
            # see FLIGHT_CLEARANCE_EPS's own comment).
            params={"command_name": "base_velocity", "height_threshold": FLIGHT_CLEARANCE_EPS},
        )
        # v7 fix (2026-08-19, claude-tg-base's review): was -3.0 -- see the
        # function's own docstring for the sign-inversion this caused (idle
        # stability penalized, contact chatter effectively rewarded). Positive,
        # matching Atanassov's own go1_config.py `change_of_contact = 10.0`.
        self.rewards.jump_change_of_contact = RewTerm(
            func=jump_change_of_contact,
            weight=3.0,
            params={"command_name": "base_velocity"},
        )

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeB2JumpRoughEnvCfg":
            self.disable_zero_weight_rewards()


# v7 Stage 1 (2026-08-19, Atanassov two-stage curriculum, "ignition через
# vertical проще" -- their own go1_upwards_config.py vs go1_config.py diff,
# checked directly): pure vertical jump-in-place, no horizontal command at
# all. Reuses UnitreeB2JumpRoughEnvCfg's entire command/phase/reward
# machinery unmodified except the sampled direction set -- with
# directions=((0.0,0.0),), jump_flight_distance/jump_direction_velocity fully
# zero (each multiplies by a body-frame `direction` dot product that is now
# always (0,0)), so there is nothing to separately retire for those two.
# CORRECTION (claude-tg-base's review): jump_direction_precision only PARTLY
# zeroes -- its v_perp term depends on `direction` the same way and goes to
# zero, but its `w_z` (yaw-rate) term is added unconditionally, independent
# of direction (see the function's own body) -- it stays live and still
# penalizes body twist during a vertical launch, which is exactly what a
# pure-vertical jump should want anyway (no reason to spin). Verified via the
# stage-1 smoke test: jump_flight_distance/jump_direction_velocity read
# 0.0000 every iteration; jump_direction_precision does too here only
# because the random early policy barely yaws yet, not because the term is
# structurally zero.
#
# Deliberately NOT a full port of their own stage-1 config: their version
# also shortens episode_length_s (4->3), reduces "gentleness" weights across
# the board (task_pos 1500->200 etc. -- inapplicable here anyway, this file
# has no task_pos/ori since it never had a landing-POSITION command to begin
# with), adds has_jumped_random_prob-style episode-init domain randomization,
# and toggles continuous_jumping off (reset after every single jump). That
# last piece in particular does not map cleanly onto this file's own
# "one long episode, many jump cycles" structure without a new termination
# term -- scoped out for this round (documented simplification, not an
# oversight); flagged as a candidate follow-up if Stage 1 alone doesn't
# ignite.
@configclass
class UnitreeB2JumpUpwardRoughEnvCfg(UnitreeB2JumpRoughEnvCfg):
    """Stage 1 of the v7 two-stage curriculum: vertical-only jump-in-place,
    meant to be trained first and short (ignition), with Stage 2 (forward,
    full weights) resuming from whatever checkpoint this produces."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.directions = ((0.0, 0.0),)

        if self.__class__.__name__ == "UnitreeB2JumpUpwardRoughEnvCfg":
            self.disable_zero_weight_rewards()
