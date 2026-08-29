# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Our own walking config, built on top of robot_lab's stock UnitreeB2RoughEnvCfg
(2026-08-13, user: "их ходит на нашем стенде неправильно... собака семенит и
подпрыгивает" -- their walking policy takes tiny, rapid, bouncy steps on the bench).

Root cause, found by diffing B2's own rough_env_cfg.py against Go2's own (same repo,
same base class, both from upstream fan-ziqi) -- ALL of the reward terms that shape
STEPPING RHYTHM are zeroed out for B2, while Go2's own copy has them active:

    term                  B2 (stock)    Go2 (stock)
    feet_air_time              0            0.1     -- minimum swing duration
    feet_air_time_variance   (unset=0)     -1.0      -- regular cadence across feet
    feet_slide                 0           -0.1      -- no dragging while planted
    feet_gait                  0            0.5      -- diagonal-pair trot rhythm
    upward                    3.0           1.0      -- B2 3x stronger flat-orientation

With NO minimum air-time, no gait-pair synchronization, and no slide penalty, nothing
prices how a step happens -- the cheapest way to satisfy velocity tracking becomes
tiny, high-frequency, arrhythmic steps (a real trot never has to fully commit to a
stride) -- the master free-variable lesson, same class of bug diagnosed repeatedly
across jump/rear_stand/leg_lift this project, just never caught for walk because
nobody had bench-compared it against a real robot's gait before. The 3x-stronger
`upward` on top of that fights the natural pitch/bounce of a real trot, plausibly
contributing to the "подпрыгивает" bounce on top of the "семенит" mincing.

Fix: port Go2's own already-validated gait weights (not guessed values -- Go2 is the
same repo's most mature quadruped, its numbers already work on real Go2 hardware per
this repo's own history) onto B2's own foot-body names (already correctly set up in
UnitreeB2RoughEnvCfg -- `.*_calf` foot_link_name, FL_calf/RR_calf + FR_calf/RL_calf
synced pairs -- just never turned on).

Deliberately a SEPARATE file/class from UnitreeB2RoughEnvCfg, not an edit to it:
jump/rear_stand/leg_lift each explicitly document relying on feet_air_time/feet_gait/
feet_slide staying at rough's own zero default ("not a periodic gait... nothing to
retire there") -- turning these on in the shared base would silently change their
training economics too. Same isolation discipline as rear_stand's own action_rate_l2
override: confine the blast radius to the one task that actually needs the change.

Mass: inherited for free. UNITREE_B2_CFG (robot_lab/assets/unitree.py) points at the
same b2_description.urdf already rescaled to the real measured weight (73.55kg,
2026-08-10) -- every B2 task in this repo, including this one, trains on the correct
mass with no separate action needed.
"""

import math

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeB2RoughEnvCfg

# Copied literally from leg_lift_env_cfg.py's own TRACKING_SIGMA -- see
# walk_feet_lift_height's own docstring/registration comment for the full
# derivation and base's numeric re-verification.
WALK_FOOT_HEIGHT_SIGMA = 0.01  # m^2

# 2026-08-24 night, phase-clock design (base's architecture, after the
# literature-restoration probe (feet_air_time 0.1->3.0) plateaued at
# feet_air_time~-0.37/feet_gait~0.40 for hours with no further movement --
# see TRAINING_STATE.md ~22:30-22:35 for the full diagnosis chain). Ported
# from ~/lib/basic-locomotion-isaaclab (IIT-DLS-Lab), whose own B2 config
# (b2_env_cfg.py:359-361) carries these exact validated numbers -- not
# guessed. STEP_FREQ=1.4Hz, DUTY_FACTOR=0.65 (each foot in stance 65% of a
# cycle, swing 35%). PHASE_OFFSET is [0.0, 0.5, 0.5, 0.0] for FL/FR/RL/RR --
# the 0/0.5 split is a symmetric trot, fully described by ONE scalar master
# phase (FR/RL = master+0.5 mod 1, RR = master), no need for 4 independent
# clocks. This offset pairing matches our own already-active
# synced_feet_pair_names=(("FL_calf","RR_calf"),("FR_calf","RL_calf")) in
# rough_env_cfg.py -- same diagonal-pair structure, independently arrived at.
WALK_GAIT_STEP_FREQ = 1.4  # Hz
WALK_GAIT_DUTY_FACTOR = 0.65
WALK_GAIT_FEET_ORDER = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"]
WALK_GAIT_PHASE_OFFSET = torch.tensor([0.0, 0.5, 0.5, 0.0])  # matches WALK_GAIT_FEET_ORDER


class GaitPhaseCommand(CommandTerm):
    """Continuous per-env trot phase-clock, NOT a resampled goal like
    JumpPulseCommand -- this is closer to DLS-lab's own env-level
    `self._phase_signal` (locomotion_env.py:74-78,270-272), ported into
    IsaacLab's CommandTerm idiom for consistency with our own codebase
    (JumpPulseCommand is the precedent for per-env state + reset() here,
    even though the CYCLE semantics differ).

    `resampling_time_range` controls how often the base class's own
    timer-driven resample fires (see GaitPhaseCommandCfg's own comment for
    the current value + rationale -- this changed 2026-08-25 night from
    "never mid-episode" to "every few seconds", see below).

    `_update_command` runs every single env step (IsaacLab's
    CommandManager.compute() calls it unconditionally, confirmed by
    reading command_manager.py:151-166) -- this is where the master phase
    actually advances, independent of the resample timer.
    """

    cfg: "GaitPhaseCommandCfg"

    def __init__(self, cfg: "GaitPhaseCommandCfg", env) -> None:
        super().__init__(cfg, env)
        self.phase = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        # sin/cos encoding, not raw phase -- avoids the 1.0->0.0 wrap
        # discontinuity a raw scalar would hand the policy every cycle.
        angle = 2.0 * math.pi * self.phase
        return torch.stack([torch.sin(angle), torch.cos(angle)], dim=-1)

    def _update_metrics(self):
        # Cheap TensorBoard liveness signal, nothing decision-driving --
        # same role as JumpPulseCommand's own window_active_ratio.
        self.metrics["phase_mean"] = self.phase

    def _resample_command(self, env_ids):
        self.phase[env_ids] = torch.rand(len(env_ids), device=self.device)

    def _update_command(self):
        # 2026-08-25 night, idle-freeze fix (owner's own live bench check on
        # test_walk caught this -- see TRAINING_STATE.md ~23:00: standing at
        # zero command, the body visibly shakes side to side, ~3-5deg roll/
        # pitch, rhythmic). Root cause: this method ran UNCONDITIONALLY every
        # step regardless of command -- the phase kept cycling at idle even
        # though every phase-dependent reward term (walk_periodic_contact_
        # suggestion, walk_pair_match) is ALREADY gated on should_move and
        # returns exactly 0 at idle. Nothing in training ever rewarded or
        # penalized ANY specific idle response to a continuously-changing
        # phase input -- the network's idle behavior was simply whatever fell
        # out of weight-sharing with the command-active regime, unconstrained.
        # Fix: freeze the phase itself (not just the downstream rewards) when
        # should_move is false, using the SAME >0.1 threshold already used
        # throughout this file. On command resume, phase continues from
        # wherever it was frozen -- no reset, no discontinuity, matches the
        # "clock, not an event" framing this whole command term was built on.
        should_move = torch.norm(self._env.command_manager.get_command("base_velocity")[:, :2], dim=1) > 0.1
        self.phase = torch.where(
            should_move,
            (self.phase + self._env.step_dt * WALK_GAIT_STEP_FREQ) % 1.0,
            self.phase,
        )


@configclass
class GaitPhaseCommandCfg(CommandTermCfg):
    class_type: type = GaitPhaseCommand
    # 2026-08-25 night, base's mid-episode phase-jitter hypothesis (PREPARED,
    # NOT YET LAUNCHED as of writing -- GPU still busy with the walk_pair_match
    # probe, and base explicitly recommended a deliberate decision over a
    # rushed 95+h-session patch). Diagnosis chain (TRAINING_STATE.md
    # ~21:25-22:10): multi-point measurement on the SAME checkpoint proved
    # true diagonal-pair simultaneity is CAUSALLY phase-dependent (bit-for-bit
    # reproducible per start_phase, not measurement noise) -- some phase
    # starts (~0.4-0.6) give real, stable ~25-30% simultaneity all episode
    # long, others (~0.0/0.2/0.8) start comparably well but SMOOTHLY,
    # continuously decay to near-zero within ~10s (no discrete stumble found
    # at second-resolution -- base_z/roll/pitch all drift smoothly, no jump).
    # base's read: the MLP policy is memoryless (no recurrence) -- action is
    # a pure function of (physical state, phase) with no notion of "just
    # reset" vs "long-running". Even though phase itself IS already sampled
    # uniformly on every episode reset (torch.rand in _resample_command,
    # confirmed), the SPECIFIC "fresh idle pose + this phase value" input
    # combination occurs only ONCE per ~28-cycle, 20s-ish episode -- rare
    # relative to total step volume even under uniform phase coverage,
    # opening a rich-get-richer gap where an early lucky first-swing for one
    # phase region gets reinforced far more than an unlucky one elsewhere.
    #
    # Fix (pure domain randomization, NO reward-formula change): resample
    # phase to a fresh random value every few seconds DURING a live episode,
    # not just at full episode reset -- forces the policy to practice
    # "sudden phase jump + resync from current physical state" many times
    # per episode instead of once, directly upweighting exactly the scenario
    # that's currently undertrained. Mechanism note: NOT the same thing as
    # DLS-lab's own commented-out jitter (locomotion_env.py:504) -- theirs
    # is a one-time offset added at reset time (avoiding exact env-to-env
    # phase sync at spawn), not periodic mid-episode re-randomization.
    #
    # Implementation is a ONE-LINE config change, zero new code: the
    # CommandTerm base class already resamples automatically whenever
    # `time_left <= 0` (command_manager.py:151-166, confirmed by reading the
    # framework source back when this class was first written) -- setting
    # this back to a finite few-second range (from the "practically never"
    # 1e6 used for the whole night's earlier probes) makes the framework
    # call _resample_command (which already does `torch.rand(...)`) on its
    # own, periodically, no new logic needed.
    # 2026-08-26 06:4x, REVERTED back to "practically never" -- base caught
    # a confound I missed: I killed the phase-jitter EXPERIMENT's training
    # PROCESS (PID 3256177) when the owner's bench check reframed priorities
    # to crawl/idle, but never reverted this CODE line. Every checkpoint
    # since (idle-freeze fix, base_height_l2 revival at -8.0, and all three
    # escalation attempts -8.0/-12.0/-16.0 tonight) trained with mid-episode
    # phase re-randomization every 3-5s ACTIVE and UNACKNOWLEDGED -- an
    # uncontrolled variable layered under every single one of tonight's
    # weight experiments, including the -8.0 "known long clean run" cited in
    # the base_height_l2 comment above (that citation is now suspect too --
    # there is NO jitter-free data point anywhere in this lineage). base's
    # hypothesis: periodic phase re-randomization is a discrete "teleport"
    # the policy must resync from; an unlucky teleport landing on a bad
    # physical moment (especially fighting a strong height-anchor pulling
    # against the coordination terms) could itself be the real trigger for
    # the repeated Loss/learning_rate floor-crashes, not the anchor weight's
    # magnitude per se -- consistent with crash distances being "same order,
    # not identical" (670/1100/1150 iterations: depends on how many jitter
    # events accumulated and which one got unlucky, not raw iteration count
    # since restart). Reverting to isolate this variable before drawing any
    # further conclusion about base_height_l2's own safe weight.
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    asset_name: str = "robot"
    debug_vis: bool = False


def walk_periodic_contact_suggestion(env, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Ported near-literally from DLS-lab's periodic_contact_suggestion
    (custom_rewards.py:544-554) -- base's diagnosis (2026-08-24 ~22:20):
    this is the actual gradient source the 4 retired bootstrap terms above
    and the literature-restoration probe both lack. It doesn't wait for a
    liftoff to already be happening (event-triggered feet_air_time) or
    reward a frozen state that happens to match a target (swing_peak/v1-v4,
    all rejected as "constant reward regardless of action" -- see
    TRAINING_STATE.md ~22:30) -- it compares EVERY step's actual contact
    state against the phase-clock's prescribed schedule and pays/penalizes
    the match, so a foot frozen in contact past its scheduled swing window
    is WRONG every single step, not just unrewarded.

    contact-detection formula is IDENTICAL to our own feet_slide's own
    `net_forces_w_history[...].norm(dim=-1).max(dim=1)[0] > 1.0`
    (rewards.py:563) -- no new sensor convention introduced, just reused.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    )

    master_phase = env.command_manager.get_term(command_name).phase
    offset = WALK_GAIT_PHASE_OFFSET.to(master_phase.device)
    foot_phase = (master_phase.unsqueeze(1) + offset.unsqueeze(0)) % 1.0
    should_be_stance = foot_phase < WALK_GAIT_DUTY_FACTOR

    should_move = torch.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1) > 0.1
    match = torch.where(should_be_stance, is_contact.float(), (~is_contact).float())
    return torch.sum(match, dim=1) / 4.0 * should_move.float()


def walk_pair_match(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """2026-08-25 night, base's design after proving GaitReward is
    STRUCTURALLY blind to simultaneity (not a tuning problem -- see
    TRAINING_STATE.md ~18:44-19:00): GaitReward's se=clip(diff**2,
    max_err**2) compares current_air_time/current_contact_time as VALUES
    at a moment, so it can never distinguish "both feet airborne RIGHT NOW"
    from "feet take turns airborne for a similar duration" -- confirmed
    numerically (measure_gait_desync.py against the max_err=10/std=40
    probe's own final checkpoint): true FL+RR/FR+RL simultaneous-airborne
    fraction stayed at 0.0%/1.4% even after feet_gait's own reward climbed
    to near its ceiling under the widened clip. GaitReward reverted to
    stock (see registration below) -- not worth tuning further, replaced
    here with a direct, transparent instantaneous check using the SAME
    is_contact/should_move machinery as walk_periodic_contact_suggestion
    (reused, not a new unfamiliar mechanism).

    For each synced pair (FL/RR, FR/RL -- same pairing as
    WALK_GAIT_PHASE_OFFSET's own 0/0.5/0.5/0 trot split), reward 1.0 for
    that pair (0.5 of the total) when BOTH feet share the same is_contact
    state RIGHT NOW (both grounded or both airborne), 0.0 when split.
    Gated by the same should_move as walk_periodic_contact_suggestion.

    KNOWN FLOOR RISK (base's own explicit warning, same class as v1-v4's
    swing_peak-alone floor problem): a pair that never moves at all (both
    feet permanently grounded) trivially scores 1.0 forever -- this term
    ALONE cannot distinguish "good synchronized trot" from "frozen
    standing". Safe only because it runs ALONGSIDE
    walk_periodic_contact_suggestion, which independently and continuously
    penalizes standing still against the phase-clock's own schedule. If
    this term's reward climbs while walk_periodic_contact_suggestion falls
    toward zero in the same trend, that is this exact exploit's signature
    -- lower this term's weight relative to the other, don't add a new
    gate to this function itself (one probe, one lesson, per this
    project's whole night of discipline).
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    )
    # WALK_GAIT_FEET_ORDER = ["FL_calf", "FR_calf", "RL_calf", "RR_calf"] (preserve_order=True
    # on sensor_cfg.body_names) -- indices 0/1/2/3, same pairing as WALK_GAIT_PHASE_OFFSET.
    pair_fl_rr = is_contact[:, 0] == is_contact[:, 3]
    pair_fr_rl = is_contact[:, 1] == is_contact[:, 2]
    match = (pair_fl_rr.float() + pair_fr_rl.float()) / 2.0

    should_move = torch.norm(env.command_manager.get_command("base_velocity")[:, :2], dim=1) > 0.1
    return match * should_move.float()


# v1/v2/v3 history (all tried and retired, 2026-08-24 -- kept as a comment,
# not code, per this project's "log the dead ends so nobody repeats them"
# discipline). v1 rewarded ANY upward foot velocity, gated only on active
# command -- climbed (3.74->4.15 over ~11% of a probe) while feet_air_time
# stayed flat-to-worse: the policy found it cheaper to VIBRATE a foot's
# vertical velocity under full standing load (never breaking contact) than
# to step. v2 added a partial-unloading gate (contact force < a measured
# threshold, not a binary contact check) to close that -- SAME pattern
# repeated (bootstrap 2.64->3.01, feet_air_time -0.019->-0.021 over the next
# ~15%): an instantaneous quantity (velocity, or a momentary force dip from
# weight-shifting between diagonal pairs) can still be gamed by oscillation
# that never produces a real, HELD displacement. v3 switched to ABSOLUTE
# WORLD-Z height (copied from leg_lift_foot_height) -- fixed the instant-vs-
# state problem, but introduced a NEW exploit: raw world_pos_w is not
# body-relative, so the policy found it cheaper to raise/lower the WHOLE
# BODY to bring all 4 calf world-Z near the single shared target than to
# actually swing any leg -- caught live via feet_height_body (a body-frame,
# already-immune-by-construction stock term) degrading in lockstep with v3's
# own reward improving (-0.20 -> -0.555) while feet_air_time stayed dead --
# proof the legs never actually moved relative to the body.


def walk_feet_lift_height(env, command_name: str, asset_cfg: SceneEntityCfg, target_z: float) -> torch.Tensor:
    """v4 (base's synthesis after v1-v3): dense per-step bootstrap, exp-
    tracking of each foot's BODY-RELATIVE Z toward a modest swing-height
    target -- takes v3's strong side (positive, dense, ABSOLUTE-STATE
    exp-anchor with no velocity/force gate, so it can bootstrap exploration
    from a dead stop) and stock feet_height_body's strong side (body-frame
    coordinate via quat_apply_inverse, immune to raising/lowering the whole
    body -- that no longer moves this quantity for any single leg), while
    dropping both terms' weak points: v3's world-frame vulnerability, and
    feet_height_body's own velocity-gated circularity (its `tanh(foot_vel)`
    factor is ~0 for a foot that hasn't started moving yet, so it can only
    ever refine an already-moving gait, never ignite one from scratch --
    same class of "reward unavailable exactly when needed most" this project
    already rejected for the force-gate design in v2).

    All 4 feet summed (unlike leg_lift's single SELECTED leg -- walk has no
    per-leg targeting, every foot matters equally for a trot). Standard
    gate going forward: check THIS reward's own trend together with
    feet_height_body's (stock, still active as a separate regularizer, see
    its own weight comment below) -- if this term's reward climbs while
    feet_height_body degrades, that is the v3 exploit signature recurring
    and must be caught immediately, not just feet_air_time in isolation."""
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    root_quat = asset.data.root_quat_w
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_pos_rel = foot_pos_w - root_pos.unsqueeze(1)
    foot_pos_body = math_utils.quat_apply_inverse(
        root_quat.unsqueeze(1).expand(-1, foot_pos_rel.shape[1], -1), foot_pos_rel
    )
    foot_z_body = foot_pos_body[:, :, 2]
    active = torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    reward = torch.sum(torch.exp(-torch.square(foot_z_body - target_z) / WALK_FOOT_HEIGHT_SIGMA), dim=1)
    return reward * active.float()


@configclass
class UnitreeB2WalkRoughEnvCfg(UnitreeB2RoughEnvCfg):
    """See module docstring -- adds Go2-validated gait-rhythm shaping on top of the
    stock rough config, which trains a walking gait but never priced HOW it steps."""

    def __post_init__(self):
        # post init of parent (this pulls in the real-mass UNITREE_B2_CFG, action
        # scale, velocity-tracking rewards, etc. -- everything stock rough already
        # gets right; we're only adding what it's missing).
        super().__post_init__()

        # 2026-08-26, widened per-body mass randomization (base's design, after
        # the WALK mass-mismatch incident -- see TRAINING_STATE.md ~11:30-12:15
        # and b2_constants.py's own B2_URDF_DOG_MASS_FRACTIONS comment for the
        # full diagnosis chain). rough_env_cfg.py's own randomize_rigid_body_mass
        # only randomizes base_link, additively, over a narrow (-1.0, 3.0)kg
        # range -- the 12 leg bodies are NEVER mass-randomized at all. A WALK
        # checkpoint with a strong absolute base_height_l2 anchor (-16.0)
        # collapsed catastrophically (idle base_z 0.53m -> ~0.11-0.18m) once
        # actually verified under the real per-body mass distribution (fixed
        # the SAME night: the bench's own mass tool had a real, independent bug
        # -- uniform-scaling all 13 bodies instead of matching training URDF's
        # per-body proportions -- but re-verifying under the CORRECTED honest
        # model showed the collapse is real, not a measurement artifact). The
        # two official Unitree assets (bench MJCF vs training URDF) disagree on
        # thigh mass by ~57% (7.4554kg vs 4.743kg) -- a real, measured spread,
        # not a guess -- so (0.7, 1.3) multiplicative on ALL 13 dog bodies
        # (base_link + all 12 leg links, not just base_link) is the starting
        # randomization range: doesn't cover the full 57% extreme, but is
        # substantially wider than the near-zero status quo and grounded in
        # this session's own measured discrepancy. Widen further later if this
        # isn't enough (checked via the same honest _apply_dog_mass tool).
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [
            self.base_link_name
        ] + [f"{leg}_{part}" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf")]
        self.events.randomize_rigid_body_mass.params["mass_distribution_params"] = (0.7, 1.3)
        self.events.randomize_rigid_body_mass.params["operation"] = "scale"

        # 2026-08-26, WALK redesign after the mass-mismatch/bench-mismatch saga
        # (see TRAINING_STATE.md ~19:00-19:30 for the full diagnosis chain,
        # base's design, WALK_GATE_SPEC.md's own #10-per-magnitude update for
        # the evidence). Full 18s gate-harness run (walk_gate_harness.py,
        # honest per-body mass) on the just-finished checkpoint found command-
        # tracking (#10) FAILS sharply at vx=1.0 (actual=-0.06!) while vx=0.3/
        # 0.6 both PASS cleanly -- and the SAME failure magnitude (error_vel_xy
        # ~1.0) already existed on the pre-tonight 25993 donor, so this is not
        # a regression, it's a long-standing, TensorBoard-masked defect (exp-
        # kernel aggregate reward averages across ALL commanded magnitudes,
        # including trivially-tracked small ones, hiding the high-end
        # collapse).
        #
        # Root cause (base's read, same class of finding as the
        # feet_air_time 30x-underweight discovery): rough_env_cfg.py:158
        # unconditionally disables the STOCK robot_lab velocity curriculum
        # (`self.curriculum.command_levels = None`) -- WALK inherited that
        # disable, so training has ALWAYS run against the full +-1.0 m/s
        # command range from iteration 0, exactly where track_lin_vel_xy_exp's
        # exp-kernel gives a vanishing gradient for a large, expensive-to-close
        # error (same math as GaitReward's own max_err clip, found earlier
        # this same night). Re-enabling it: starts commands at 20% of the
        # full range (matching rough_env_cfg.py's own commented-out hint,
        # range_multiplier=(0.2, 1.0), not reinventing a number), expands by
        # +-0.1 m/s only once mean tracking reward exceeds 80% of
        # track_lin_vel_xy_exp's own weight on the CURRENT range -- i.e. the
        # policy only ever gets a harder command once it has honestly mastered
        # the easier one, instead of being thrown at the hardest case from day
        # one and never learning to close it (the observed vx=1.0 collapse).
        # Resuming from 25993 (warm-start) means env.common_step_counter==0
        # re-triggers at THIS run's start -- the curriculum re-validates from
        # 20% upward rather than assuming the warm-started policy already
        # earned the full range; cheap re-validation, not lost progress.
        self.curriculum.command_levels = CurrTerm(
            func=mdp.command_levels_vel,
            params={
                "reward_term_name": "track_lin_vel_xy_exp",
                "range_multiplier": (0.2, 1.0),
            },
        )

        # -- gait-rhythm shaping, ported from Go2's own validated rough_env_cfg.py
        # (see module docstring for the full B2-vs-Go2 diff and reasoning).
        #
        # 2026-08-24 night (owner's direct order to check the ORIGINAL source,
        # not guess): 0.1 was Go2's own robot_lab-ported value, but Go2's port
        # ITSELF already deviates from the original foundational recipe this
        # whole stack descends from (Rudin/ETH 2021 "Learning to Walk in
        # Minutes...", locally at ~/lib/legged_gym) -- confirmed by reading its
        # base config directly: `feet_air_time=1.0` there, EQUAL to
        # `tracking_lin_vel=1.0` (100% ratio). Our own (and Go2's own)
        # track_lin_vel_xy_exp=3.0 vs feet_air_time=0.1 is a 30x-weaker ratio
        # (3.3%) -- not something introduced tonight, a pre-existing drift in
        # robot_lab's whole port, only surfaced now because it's the exact
        # mechanism this run needed. The underlying reward FORMULA/threshold
        # is verified IDENTICAL to the original (`(air_time-0.5)*first_contact`,
        # read directly from legged_robot.py's own _reward_feet_air_time) --
        # this is a pure weight restoration, not a new mechanism. Matches
        # base's diagnosis exactly: feet_gait (state-based, lives in the
        # stochastic training distribution, immune to static-camping by
        # requiring genuine alternation) shows real-but-short lifts already
        # happening (~0.42 of its own 0.5 ceiling) -- feet_air_time's own
        # event-triggered penalty on those SAME short lifts was just 30x too
        # weak to pull the policy from "short, penalized" to "long, rewarded".
        # Structurally safer than any of tonight's 4 custom bootstrap attempts
        # too: feet_air_time only fires on a first_contact TRANSITION (a foot
        # that never leaves contact never triggers it at all) -- a frozen
        # static pose is not just unrewarded here, the term is architecturally
        # incapable of firing for one, unlike a dense absolute-state term.
        # Set to 3.0 -- literally matching track_lin_vel_xy_exp (not a softer
        # intermediate value), the exact 1:1 ratio the original source uses,
        # since we already have a validated number and no reason to guess
        # something softer.
        self.rewards.feet_air_time.weight = 3.0
        # 2026-08-25, threshold-conflict fix (base's diagnosis, confirmed by the
        # full phase-clock probe result -- TRAINING_STATE.md ~04:50-05:00,
        # TRAIN_RESEARCH.md's own phase-clock entry): feet_air_time's threshold
        # stayed at rough_env_cfg.py's stock 0.5s while phase-clock's own
        # duty_factor=0.65 @ 1.4Hz PRESCRIBES a ~0.25s swing -- exactly half the
        # threshold. The policy spent the whole phase-clock probe (it10998->
        # 16996, ~8h combined) optimizing walk_periodic_contact_suggestion
        # (1.29->1.51, ~78% of its own ceiling, genuinely learning the schedule)
        # while feet_air_time stayed dead flat (-0.29..-0.32 the entire time) --
        # every correctly-scheduled short swing paid (0.25-0.5)=-0.25 here, a
        # structural, not incidental, conflict between the two terms over the
        # same underlying quantity (swing duration). Lowered to 0.2s -- close to
        # phase-clock's own prescribed duration, so a swing that matches the
        # schedule stops being penalized as "too short" by this term too.
        self.rewards.feet_air_time.params["threshold"] = 0.2
        self.rewards.feet_air_time_variance.weight = -1.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_gait.weight = 0.5
        # 2026-08-25, gait-synchrony clip-and-scale widen (base's design, PHASE 1
        # of a two-step plan -- see this comment's own "revert" note below).
        # WALK_GATE_SPEC.md's pattern-8/9 root gate (diagonal synchrony, all-4-
        # leg balance) is still failing after the feet_air_time.threshold fix
        # above -- measured (measure_gait_desync.py, scratchpad) real air_time/
        # contact_time desync between GaitReward's synced pairs at 8.5-8.8s mean,
        # up to 18.6s max (94-97% of samples beyond stock max_err=0.2's clip
        # boundary -- see rewards.py:237-244's own
        # `se=clip(square(diff),max=max_err**2)`, `reward=exp(-se/std)`). Simply
        # widening max_err alone (my own first, WRONG instinct, corrected before
        # applying) does nothing: at max_err=8-10 with stock std=sqrt(0.5), the
        # EXPONENT already kills the signal (exp(-64/0.707)~0 at diff=8s) before
        # the clip boundary is ever reached -- max_err and std must widen
        # TOGETHER, verified by hand at several diff values (both by base and
        # independently re-checked here): max_err=10, std=40 gives diff=0->
        # reward=1.0, diff=8.5(today's mean)->0.164, diff=10(clip edge)->0.082,
        # diff=18(today's max)->0.082 (same, clipped, expected), diff=3(partial
        # progress)->0.80, diff=0.5(near final target)->0.994 -- a real,
        # non-degenerate gradient across the WHOLE distance from today's mean to
        # the eventual target, not just near-perfect synchrony. Applies to BOTH
        # sync_reward and async_reward (GaitReward uses one shared max_err/std
        # for both, not separate params -- confirmed reading __call__/_sync_
        # reward_func/_async_reward_func).
        #
        # REVERT PLAN (do not skip): once feet_gait shows real upward movement
        # (confirming synchrony gradient is working), max_err/std must be
        # narrowed back toward stock (0.2/sqrt(0.5)) or an intermediate value in
        # a SEPARATE later structural step -- loose max_err/std forever would
        # reward approximate-but-not-real synchrony, not the tight diagonal trot
        # WALK_GATE_SPEC.md's pattern 8 actually requires. One variable at a
        # time: this probe widens ONLY max_err/std, nothing else changes.
        #
        # RESULT (2026-08-25 ~18:21-19:00, ran to completion): feet_gait DID
        # climb to near its own ceiling (0.45->0.47 of 0.5) -- verified NOT a
        # cheap mechanical artifact (computed sync_reward on the identical
        # trajectory under old vs new params: new params give LOWER reward,
        # 0.24 vs 0.77, proving the climb is real policy adaptation, not free
        # reward from the parameter change itself). BUT direct numeric check
        # (measure_gait_desync.py, counting actual simultaneous-airborne
        # events) found true FL+RR/FR+RL BOTH-airborne-at-once fraction stuck
        # at 0.0%/1.4% even at this checkpoint -- the policy adapted toward
        # "similar current_air_time/current_contact_time VALUES", exactly
        # what GaitReward's se=clip(diff**2,...) formula actually measures,
        # NOT toward genuine simultaneity. This is a STRUCTURAL blindness of
        # the formula itself (it compares state durations, never checks "are
        # both feet in the SAME state right now") -- no (max_err, std) choice
        # can fix this, confirmed by base independently re-deriving the same
        # conclusion. Reverted to stock below; replaced with a direct
        # instantaneous-match term (walk_pair_match, this file, module level)
        # that checks is_contact equality between paired feet AT THE CURRENT
        # STEP, the thing GaitReward can never see.
        self.rewards.feet_gait.params["max_err"] = 0.2
        self.rewards.feet_gait.params["std"] = math.sqrt(0.5)
        self.rewards.walk_pair_match = RewTerm(
            func=walk_pair_match,
            weight=1.5,  # base's first guess, same order as walk_periodic_contact_suggestion's own
                         # 2.0 -- flag as tunable, not validated. KNOWN FLOOR RISK: this term alone
                         # trivially rewards a pair that never moves (both permanently grounded) --
                         # safe only alongside walk_periodic_contact_suggestion's own independent
                         # standing-still penalty. Watch for the exploit signature: this term rising
                         # while walk_periodic_contact_suggestion falls toward zero in the same
                         # trend -- see walk_pair_match's own docstring for the full warning.
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=WALK_GAIT_FEET_ORDER, preserve_order=True),
            },
        )

        # 2026-08-25 night, base_height_l2 revival (base's find after the owner's
        # live bench check on test_walk caught "приседает и ползёт" -- squats and
        # crawls instead of walking, TRAINING_STATE.md ~23:00). rough_env_cfg.py:90
        # sets `base_height_l2.weight = 0` -- this stock term is INHERITED, ZEROED,
        # and walk_env_cfg.py never touched it, this whole project's history (base
        # read the full diff to confirm). `target_height=0.53` is ALREADY configured
        # (rough_env_cfg.py:91, lands mid-range of the StandFix anchor 0.50-0.56m)
        # -- only the weight was ever missing. NOT a new recipe: crawl_env_cfg.py and
        # vision_env_cfg.py both already use this exact term at weight=-8.0 (crawl
        # with a different, deliberately-low target=0.35 for its own crouched gait).
        # Root-cause read: `feet_height_body` (the term monitored all night as "under
        # observation" while it drifted -0.23->-0.30) is a FOOT-relative metric --
        # correlates with squatting but is not the same signal as root base_z, and
        # was never a substitute for directly holding body height. Measured directly
        # (scratchpad, this session): base_z idle=0.5477m -> forward-command
        # mean=0.3707m (min=0.3254m) on the pair_match final checkpoint, WITH
        # substantial horizontal foot velocity (mean 1.0-1.2 m/s) -- ruled out the
        # "pure vertical squat, feet_height_body's own tanh(horizontal_vel) gate
        # goes blind" exploit class from earlier tonight (v3's own diagnosis) --
        # this is real forward-shuffling at a collapsed height, not a frozen-squat
        # trick, consistent with the owner's own "crawls" description. Re-enabling
        # this stock, already-validated-elsewhere term is a re-enable of existing
        # machinery, not a new experimental variable -- same logic as reverting
        # GaitReward to stock and re-launching pair_match together, the night before.
        # 2026-08-26, escalation (base's design + explicit GO after 3 consecutive
        # measurement points, TRAINING_STATE.md ~01:03-01:10): -8.0 plateaued --
        # forward-command base_z held flat at mean~0.40m (target 0.53m) across
        # it26300/26700/27100, and worst-case MIN kept monotonically worsening
        # (0.3857->0.3538->0.3352, no hint of reversal, unlike coordination
        # terms' own dip-then-recover pattern earlier that night) -- a real,
        # not-yet-reversed trend, not noise. base's biomechanical read: -8.0
        # fights a genuine physical trade-off (lower CoM = more stability margin
        # during coordinated pair-swing, which pair_match/periodic_contact pay
        # 1.5-2.0 for) that crawl_env_cfg.py's own -8.0 never had to fight
        # (crawl WANTS low, nothing opposes it there). Proportionate escalation
        # (doubling, not a wild jump) -- same staged-escalation idiom as JUMP's
        # own hip_neutral/flight_symmetry weight history. MUST re-verify idle
        # stability (currently excellent, roll/pitch <2 deg) doesn't regress
        # with a stronger anchor -- not assumed safe just because idle-freeze
        # gates the coordination terms separately.
        # 2026-08-26, de-escalation to -12.0: -16.0 showed a REPLICATED PPO
        # destabilization pattern -- Loss/learning_rate (rsl_rl's adaptive
        # KL scheduler) crashed to its floor (1e-5) roughly ~1000-1150
        # iterations into training under this weight, independently in TWO
        # separate lineages (original continuous run: it27100->28100-28250;
        # fresh-optimizer restart from it27900: it27900->28900-29000, same
        # ~1000-1100 iteration distance). Both times mean_reward/base_height_l2
        # still LOOKED healthy for a while after the LR crash (near-zero LR
        # means the policy barely updates, masking the damage briefly), which
        # is why the first restart attempt (from a checkpoint already just
        # past the crash point) misleadingly looked fine at first read. Two
        # independent occurrences at a similar time-constant rules out "one
        # bad batch" and points to -16.0 itself being too aggressive relative
        # to what the optimizer's adaptive KL trust region can absorb over
        # sustained training, not a fluke. Halving the escalation delta
        # (16->12, i.e. same +4.0 step size as -8.0->-12.0 would have been)
        # instead of reverting fully to -8.0 -- -8.0 itself was already
        # plateauing (see comment above) before proportionate escalation was
        # chosen, so -12.0 is the next proportionate step, not a full retreat.
        # 2026-08-26, THIRD occurrence, now under -12.0: same
        # sustained-LR-floor pattern recurred again (it29470+ onward, ~670
        # iterations after this weight took effect at it28800) -- distinct
        # from the transient reward noise seen at it28900-29100 right after
        # the -16.0->-12.0 switch (that one was isolated single-step dips
        # with immediate full recovery every time; this one is Loss/
        # learning_rate PERSISTENTLY pinned at 1e-5 with no oscillation back
        # up, plus a degraded reward baseline -- the same qualitative
        # signature as both prior -16.0 crashes). Three independent
        # occurrences (two at -16.0, one at -12.0) at varying but broadly
        # similar iteration distances (670-1150) raises the real
        # possibility this isn't purely about this weight's magnitude --
        # could be an environment/curriculum-driven periodic event
        # (terrain difficulty ramp, rare outlier trajectory) that any
        # nonzero base_height_l2 anchor happens to interact badly with.
        # Reverted to -8.0 and relaunched (PID 708703, still running as a
        # parallel data point -- cheap to let finish, see below for why it's
        # no longer a clean baseline either).
        #
        # 2026-08-26 ~06:5x, base caught a confound BEFORE this -8.0 result
        # could be trusted: GaitPhaseCommandCfg.resampling_time_range (this
        # file, ~30 lines below) was STILL (3.0, 5.0) -- the phase-jitter
        # EXPERIMENT's training process was killed hours ago, but the CODE
        # was never reverted. Every checkpoint since idle-freeze/height-
        # revival began (including the -8.0 "known clean run" cited above)
        # trained with uncontrolled mid-episode phase re-randomization. See
        # the resampling_time_range comment for the full hypothesis (a bad-
        # luck jitter "teleport" landing on a rough physical moment, not
        # weight magnitude, may be the real crash trigger).
        #
        # Reverted jitter to "practically never" AND restored the weight to
        # -16.0 (best height result before any crash) for a CLEAN control:
        # this isolates whether removing jitter alone fixes stability at the
        # most aggressive weight tried. If it holds past ~1200 iterations
        # (further than any of tonight's three jitter-confounded attempts),
        # jitter was the real cause and -16.0 is usable. If it still crashes,
        # the weight-magnitude hypothesis stands independent of jitter.
        self.rewards.base_height_l2.weight = -16.0
        # walk_feet_lift_height's own module-level + docstring comments for
        # the full diagnosis chain). v4 = v3's formula (positive, dense,
        # absolute-STATE exp-anchor, no velocity/force gate -- can ignite
        # from a dead stop) applied to a BODY-RELATIVE coordinate (same
        # quat_apply_inverse transform stock feet_height_body already uses --
        # immune to whole-body lift/squat BY CONSTRUCTION, not a patched-on
        # penalty: a body-relative foot-Z literally cannot change from moving
        # the whole body, with or without horizontal foot velocity, so there
        # is nothing left to exploit that way, unlike a -30-weighted penalty
        # patch base explicitly rejected -- that would still fully zero out
        # via feet_height_body's own tanh(horiz_vel)=0 gate on a PURELY
        # vertical squat, the cheapest version of this exploit, regardless of
        # penalty weight).
        #
        # target_z=-0.186m: measured (not guessed) this donor's (model_7999)
        # own idle BODY-RELATIVE calf-Z via a live sensor read (quat_apply_
        # inverse(root_quat, calf_world_pos - root_world_pos), same transform
        # as this term's own code) -- FL=-0.2541 FR=-0.2695 RL=-0.2571
        # RR=-0.2834, mean=-0.2660 (spread 0.029m, notably TIGHTER than the
        # world-frame idle spread of 0.06m -- body-relative coordinates
        # partly cancel whatever per-leg world-frame asymmetry existed).
        # +0.08m above that mean (same order-of-magnitude trot-swing-
        # clearance literature figure used for v3, now applied in the
        # correct body-relative frame) -> target_z=-0.186m.
        # WALK_FOOT_HEIGHT_SIGMA=0.01 (same as v3, copied from leg_lift) is
        # live at this target: exp(-0.08^2/0.01)=0.527 at idle, 0.852
        # halfway, 0.998 at the target -- unchanged from v3's own check since
        # the delta magnitude is the same, just re-based to the body-relative
        # origin.
        #
        # Standard gate from now on: check THIS term's own reward trend
        # TOGETHER WITH stock feet_height_body's (weight -5.0, still active,
        # untouched -- see rough_env_cfg.py) -- if this climbs while
        # feet_height_body degrades, that is v3's exploit signature
        # recurring in a new shape and must be caught immediately, not just
        # feet_air_time in isolation.
        # DISABLED for this probe (2026-08-24 night, owner+base's literature-
        # restoration test) -- one variable at a time: this probe tests
        # feet_air_time's own restored weight (0.1->3.0, see above) in
        # isolation, not stacked on top of a 4th custom bootstrap version.
        # walk_feet_lift_height itself stays defined (not deleted) in case
        # the literature fix alone isn't sufficient and this needs revisiting
        # later, in a non-rushed follow-up session.
        # self.rewards.walk_feet_lift_bootstrap = RewTerm(
        #     func=walk_feet_lift_height,
        #     weight=5.0,
        #     params={
        #         "command_name": "base_velocity",
        #         "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
        #         "target_z": -0.186,
        #     },
        # )

        # 2026-08-24/25 night, phase-clock (base's design, next structural
        # fix after the literature-restoration probe's plateau -- see this
        # file's own module-level GaitPhaseCommand/walk_periodic_contact_
        # suggestion docstrings for the full mechanism + DLS-lab source).
        # ENABLED (2026-08-25): literature-restoration probe finished at
        # it10998 (donor model_10998.pt, run 2026-08-24_20-29-25) --
        # feet_air_time plateaued -0.35..-0.40 the entire second half, no
        # breakthrough (TRAINING_STATE.md full TensorBoard trend). Checkpoint
        # surgery run on the REAL final checkpoint (not the earlier synthetic
        # test) -- actor 45->47, critic 235->237 (asymmetric actor-critic,
        # critic keeps base_lin_vel+height_scan the policy doesn't get --
        # confirmed via rough_env_cfg.py's own `observations.policy.
        # base_lin_vel/height_scan = None`), both zero-init, max action/value
        # diff vs donor = 0.00e+00. base's GO already given after independent
        # code review (command_manager.py timing, step_dt calculation both
        # re-verified against base's own IsaacLab copy).
        self.commands.gait_phase = GaitPhaseCommandCfg()
        self.observations.policy.gait_phase = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "gait_phase"}, clip=(-1.0, 1.0), scale=1.0
        )
        self.observations.critic.gait_phase = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "gait_phase"}, clip=(-1.0, 1.0), scale=1.0
        )
        self.rewards.walk_periodic_contact_suggestion = RewTerm(
            func=walk_periodic_contact_suggestion,
            weight=2.0,  # first guess, comparable to feet_gait's own 0.5 ceiling scaled for a per-step (not
                         # per-transition) term -- flag to base as tunable, not a validated number like the
                         # DLS-lab source constants above it.
            params={
                "command_name": "gait_phase",
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=WALK_GAIT_FEET_ORDER, preserve_order=True),
            },
        )

        # 3.0 -> 1.0 (Go2's own value): B2's stock 3x-stronger flat-orientation pull
        # plausibly fights the natural pitch/bounce of an actual trot on top of the
        # missing gait-rhythm terms above -- both contribute to the reported bounce,
        # ease off to Go2's own already-working value rather than guessing a new one.
        #
        # 2026-08-26, 1.0 -> 2.0 (WALK redesign, base's literature research --
        # ~base/DISTILLED/2026-08-26_walk-redesign-literature.md): standard
        # practice keeps flat-orientation penalty at ~70% of the tracking
        # weight (track_lin_vel_xy_exp=3.0) -- the 1.0 above was set back when
        # WALK was first created, against foot-vibration, BEFORE any of the
        # absolute height-anchor work existed. With base_height_l2=-16.0 now
        # pulling hard on root-Z and orientation comparatively under-weighted,
        # "nose down, tail up" becomes a cheap way to formally satisfy height
        # while failing WALK_GATE_SPEC.md #2 (pitch, measured 12.4°
        # instant-max tonight, gate threshold 6°). Proportionate escalation
        # (1.0->2.0, not straight to the literature's stock 3.0-equivalent) --
        # same staged-escalation idiom used all night for base_height_l2/
        # flight_symmetry, not a leap to an unvalidated number.
        self.rewards.upward.weight = 2.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeB2WalkRoughEnvCfg":
            self.disable_zero_weight_rewards()
