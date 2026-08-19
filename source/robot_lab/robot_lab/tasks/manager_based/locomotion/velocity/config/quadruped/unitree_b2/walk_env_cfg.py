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

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeB2RoughEnvCfg


@configclass
class UnitreeB2WalkRoughEnvCfg(UnitreeB2RoughEnvCfg):
    """See module docstring -- adds Go2-validated gait-rhythm shaping on top of the
    stock rough config, which trains a walking gait but never priced HOW it steps."""

    def __post_init__(self):
        # post init of parent (this pulls in the real-mass UNITREE_B2_CFG, action
        # scale, velocity-tracking rewards, etc. -- everything stock rough already
        # gets right; we're only adding what it's missing).
        super().__post_init__()

        # -- gait-rhythm shaping, ported from Go2's own validated rough_env_cfg.py
        # (see module docstring for the full B2-vs-Go2 diff and reasoning).
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_air_time_variance.weight = -1.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_gait.weight = 0.5

        # 3.0 -> 1.0 (Go2's own value): B2's stock 3x-stronger flat-orientation pull
        # plausibly fights the natural pitch/bounce of an actual trot on top of the
        # missing gait-rhythm terms above -- both contribute to the reported bounce,
        # ease off to Go2's own already-working value rather than guessing a new one.
        self.rewards.upward.weight = 1.0

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeB2WalkRoughEnvCfg":
            self.disable_zero_weight_rewards()
