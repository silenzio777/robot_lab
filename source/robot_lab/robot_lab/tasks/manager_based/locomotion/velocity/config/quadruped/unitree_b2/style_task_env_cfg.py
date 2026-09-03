"""Style+Task гибрид (2026-09-03, дизайн base style-task-hybrid-design.md,
приказ хозяина «сделать конфиг управляемой политики IL+RL — рулить
джойстиком эту походку»): velocity-tracking задача walk_reset (v4-пакет:
живой legged_gym reset/push + base_height -8.0 + feet_air_time 3.0)
ПЛЮС style-награды «двигай ногами как клип медленного шага» поверх.

Ключевые решения (все из дизайн-дока, зафиксированы с base):
- СТИЛЬ ROOT-ОТНОСИТЕЛЬНЫЙ, БЕЗ YAW и БЕЗ мировой позиции: курс и
  скорость принадлежат ТОЛЬКО task-термам (иначе стиль воевал бы с
  джойстиком). Style сравнивает: суставные позы/скорости, высоту root,
  roll/pitch — всё индексировано круговой фазой клипа.
- Фаза: свободные часы t mod T, obs (cos, sin) ДОПОЛНИТЕЛЬНЫМИ слотами
  ПОСЛЕ стандартных 45 walk-наблюдений → obs 47. velocity_commands
  остаются на своих местах (слоты 6-8) — стендовый wrapper walk-раскладки
  работает как есть, фаза-хвост синтезируется драйвером/часами
  (командная конвенция = command_mid с хвостовой фазой; стендовая
  поддержка — отдельным коммитом robot_stand при выводе на кнопку).
- Терминации walk_reset КАК ЕСТЬ (падение и т.п.), НИКАКОГО
  imitation-diverged поводка: стиль мягкий (награда), не строгий.
- RSI-телепорта НЕТ: ресеты v4-пакета (default поза × uniform(0.5,1.5)),
  политика ОБЯЗАНА выучить вход в походку из стойки — ровно то, чего
  чистому IL не хватало для реального робота.
- Вариант C по скоростям (хозяин): клип 1.01 м/с, команды ±1.0 —
  рассогласование на краях принято, стиль всегда включён.
- num_envs 8192 (одобрено хозяином для серьёзного IL-рана).

ВЕСА/ШКАЛЫ STYLE-ТЕРМОВ — НЕ PMC-числа вслепую (урок joint_acc_l2), а
ИЗМЕРЕНЫ на ходячем роллауте it2300 (scratchpad/style_term_measure.py,
2026-09-03): joint_pos_sq mean 0.22 → scale 1.0 даёт kernel 0.81;
joint_vel_sq mean 53 → PMC-шкала 0.1 душит (kernel 0.10), взято 0.01
(kernel 0.59); root_h mean err 0.016 м → scale 20 (kernel 0.73);
roll+pitch mean 0.056 rad → scale 10 (kernel 0.57). Task-сумма
(track_lin 3.0 + track_ang 1.5) = 4.5 → style-сумма 1.8 (40%, середина
коридора 30-50% из дизайна): joint_pos 1.0, joint_vel 0.15, root_h 0.3,
root_rp 0.35.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg, ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .walk_reset_env_cfg import UnitreeB2WalkResetEnvCfg

# Клип подтверждённого скилла il_slow_walk (confirmed_skills/il_slow_walk/
# mocap_source/LINEAGE.md — полная родословная). Тот же файл, что учил
# slowwalk-политику.
STYLE_MOTION_JSON = Path(
    "/home/silenzio/lib/urdf_stand/retargeted_csv/training/dog_quad_walk_001_slowloop_wide10cm.json"
)

# --- шкалы/веса: ИЗМЕРЕНЫ, см. модульный докстринг ---
STYLE_SCALE_JOINT_POS = 1.0
STYLE_SCALE_JOINT_VEL = 0.01
STYLE_SCALE_ROOT_H = 20.0
STYLE_SCALE_ROOT_RP = 10.0
STYLE_WEIGHT_JOINT_POS = 1.0
STYLE_WEIGHT_JOINT_VEL = 0.15
STYLE_WEIGHT_ROOT_H = 0.3
STYLE_WEIGHT_ROOT_RP = 0.35

# Протокол защиты пересадки (вердикт base 2026-09-03, v6-разучивание
# 10.3->87.2 за 400 ит): 50% ресетов -- RSI-телепорт на кадр клипа
# (ходячий опыт в батче -> критик узнаёт value ходьбы за warmup ->
# walking-положительные advantages защищают навык после разморозки);
# другие 50% -- обычный v4-ресет из стойки (учат вход). Фаза и поза --
# ОДНА переменная внутри _resample_command: рассинхрон часов и кадра
# невозможен по построению (ловушка Л3 base закрыта конструктивно).
STYLE_TELEPORT_FRACTION = 0.5


class StylePhaseCommand(CommandTerm):
    """Свободные круговые фазовые часы клипа + root-относительный референс.

    В отличие от MotionRefCommand (imitation_env_cfg): НЕ телепортирует,
    НЕ хранит мировой root-путь (только высоту/roll-pitch/суставы —
    root-относительный стиль без yaw), фаза зациклена t mod T и стартует
    случайной на каждом ресете (чтобы политика не привязала фазу к
    моменту ресета)."""

    cfg: "StylePhaseCommandCfg"

    def __init__(self, cfg: "StylePhaseCommandCfg", env) -> None:
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.asset_name]

        with open(cfg.motion_json_path) as f:
            data = json.load(f)
        assert data["frame_layout"] == "root_pos_xyz(3) + root_quat_wxyz(4) + joint_pos(12)"
        assert data["root_quat_convention"] == "wxyz"
        frames = torch.tensor(data["frames"], dtype=torch.float32, device=self.device)
        self.n_frames = frames.shape[0]
        self.frame_dt = float(data["frame_dt"])
        self.duration_s = (self.n_frames - 1) * self.frame_dt

        # Полные кадры -- ТОЛЬКО для 50%-телепорта ресетов (награды стиля
        # их не видят, стиль остаётся root-относительным без yaw).
        self._tp_root_pos = frames[:, 0:3]
        self._tp_root_quat = frames[:, 3:7]
        tp_lin = torch.zeros_like(self._tp_root_pos)
        tp_lin[1:-1] = (self._tp_root_pos[2:] - self._tp_root_pos[:-2]) / (2 * self.frame_dt)
        tp_lin[0] = (self._tp_root_pos[1] - self._tp_root_pos[0]) / self.frame_dt
        tp_lin[-1] = (self._tp_root_pos[-1] - self._tp_root_pos[-2]) / self.frame_dt
        self._tp_root_lin_vel = tp_lin

        # root-относительные величины: высота, roll/pitch (из кватерниона,
        # yaw отброшен), суставы. Мировые x/y и yaw НЕ сохраняем вовсе.
        self.ref_root_h = frames[:, 2]
        quat = frames[:, 3:7]
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        self.ref_roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        self.ref_pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
        self.ref_joint_pos = frames[:, 7:19]
        # конечная разность как в imitation (клип статичен)
        jv = torch.zeros_like(self.ref_joint_pos)
        jv[1:-1] = (self.ref_joint_pos[2:] - self.ref_joint_pos[:-2]) / (2 * self.frame_dt)
        jv[0] = (self.ref_joint_pos[1] - self.ref_joint_pos[0]) / self.frame_dt
        jv[-1] = (self.ref_joint_pos[-1] - self.ref_joint_pos[-2]) / self.frame_dt
        self.ref_joint_vel = jv

        joint_ids, found = self.asset.find_joints(data["joint_order"], preserve_order=True)
        assert found == data["joint_order"]
        self.joint_ids = joint_ids

        # Per-сустав вес err_sq в style_joint_pos -- ПОСЛЕДОВАТЕЛЬНЫЕ
        # подтверждённые шаги (base+train, 2026-09-03/04), не потолочный
        # набор весов разом:
        # ШАГ 1 (thigh x2, v9 старт): живой замер v8 (thigh ROM 47-53%,
        # частота шага 3.10x клока) показал PPO дёшево сжал thigh (дорогой
        # маховый сустав). thigh x2 внесён один -- 3 из 4 ног улучшились
        # (it4200->4500), но hip (не трогали) поплыл: RL 143%->196%,
        # затем на it4800 распространилось на FL (68->125%) и вернулось на
        # RR (149->174%) -- весь hip-класс дрейфует под новой экономикой,
        # не одна нога отстаёт (довод ПРОТИВ per-leg, ЗА глобальный фикс).
        # err_sq calf/thigh на it4800 тоже просел -- дифференциальный
        # критерий на следующей точке: если thigh/calf развернутся ВМЕСТЕ
        # с hip после этого шага -- каскад от hip закрыт одним фиксом;
        # если hip один поправился, а thigh/calf остались на уровне it4800
        # -- отдельный необъяснённый фактор, не hip.
        # ШАГ 2 (hip x2, этот коммит): тот же инструмент, что thigh,
        # единообразно всем 4 ногам.
        self.joint_pos_weights = torch.tensor(
            [2.0 if ("thigh" in n or "hip" in n) else 1.0 for n in data["joint_order"]],
            dtype=torch.float32, device=self.device,
        )

        # Стопы (calf-прокси, как во всём репо) -- для терраин-инвариантной
        # высоты style_root_h (см. функцию: root_z − mean(feet_z)).
        foot_names = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
        body_ids, found_b = self.asset.find_bodies(foot_names, preserve_order=True)
        assert found_b == foot_names
        self.foot_body_ids = body_ids

        self.phase_time = torch.zeros(self.num_envs, device=self.device)

    def _lookup(self, values: torch.Tensor) -> torch.Tensor:
        """Линейная интерполяция values (F, ...) по зацикленному phase_time."""
        t = self.phase_time % self.duration_s
        frame_f = t / self.frame_dt
        i0 = frame_f.long().clamp(0, self.n_frames - 2)
        alpha = (frame_f - i0.to(frame_f.dtype)).clamp(0.0, 1.0)
        v0, v1 = values[i0], values[i0 + 1]
        if values.dim() == 1:
            return torch.lerp(v0, v1, alpha)
        return torch.lerp(v0, v1, alpha.unsqueeze(-1))

    # --- запросы для reward-термов ---
    def ref_h(self) -> torch.Tensor:
        return self._lookup(self.ref_root_h)

    def ref_rp(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._lookup(self.ref_roll), self._lookup(self.ref_pitch)

    def ref_joints(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self._lookup(self.ref_joint_pos), self._lookup(self.ref_joint_vel)

    @property
    def command(self) -> torch.Tensor:
        ang = 2.0 * torch.pi * (self.phase_time % self.duration_s) / self.duration_s
        return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)

    def _update_metrics(self):
        self.metrics["style_phase_mean"] = (self.phase_time % self.duration_s) / self.duration_s

    def _resample_command(self, env_ids):
        n = len(env_ids)
        if n == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device)
        # случайная стартовая фаза для ВСЕХ; для STYLE_TELEPORT_FRACTION
        # сред -- ещё и RSI-телепорт на кадр ЭТОЙ ЖЕ фазы (протокол защиты
        # пересадки, см. константу): фаза и поза синхронны по построению.
        start_t = torch.rand(n, device=self.device) * self.duration_s
        self.phase_time[env_ids_t] = start_t

        tp_mask = torch.rand(n, device=self.device) < STYLE_TELEPORT_FRACTION
        if tp_mask.any():
            tp_ids = env_ids_t[tp_mask]
            t = start_t[tp_mask]
            frame_f = t / self.frame_dt
            i0 = frame_f.long().clamp(0, self.n_frames - 2)
            alpha = (frame_f - i0.to(frame_f.dtype)).clamp(0.0, 1.0).unsqueeze(-1)
            root_pos = torch.lerp(self._tp_root_pos[i0], self._tp_root_pos[i0 + 1], alpha)
            root_quat = self._tp_root_quat[i0]  # nearest -- как проверенный RSI imitation
            jp = torch.lerp(self.ref_joint_pos[i0], self.ref_joint_pos[i0 + 1], alpha)
            jv = torch.lerp(self.ref_joint_vel[i0], self.ref_joint_vel[i0 + 1], alpha)
            lin = torch.lerp(self._tp_root_lin_vel[i0], self._tp_root_lin_vel[i0 + 1], alpha)

            root_pos_world = root_pos + self._env.scene.env_origins[tp_ids]
            self.asset.write_root_pose_to_sim(torch.cat([root_pos_world, root_quat], dim=-1), env_ids=tp_ids)
            root_vel = torch.cat([lin, torch.zeros_like(lin)], dim=-1)
            self.asset.write_root_velocity_to_sim(root_vel, env_ids=tp_ids)
            self.asset.write_joint_state_to_sim(jp, jv, joint_ids=self.joint_ids, env_ids=tp_ids)

    def _update_command(self):
        self.phase_time = self.phase_time + self._env.step_dt


@configclass
class StylePhaseCommandCfg(CommandTermCfg):
    class_type: type = StylePhaseCommand
    motion_json_path: Path = STYLE_MOTION_JSON
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    asset_name: str = "robot"
    debug_vis: bool = False


# --- style reward-термы (все exp-kernel, root-относительные, без yaw) ---
#
# ГЕЙТ ПО КОМАНДЕ (v5, вердикт base 2026-09-03 после трёх нулевых проб
# гибрида v4): безусловный положительный стиль оказался СУБСИДИЕЙ
# ВЫЖИВАНИЯ-КОНСЕРВАТИЗМА (патология survival bonus, семья v1-крауча --
# только статуя в полный рост: base_height доволен, стиль капает).
# Гейт СТРОГО по КОМАНДЕ (|cmd| >= 0.1), НЕ по фактическому движению --
# команду нельзя эксплуатировать, движение можно ('вибрация' ради
# выплаты). Семантика: cmd~0 -> стиль РОВНО 0 (субсидии нет); cmd>0 ->
# любой шаг платит style>0 против нуля стойки -- градиент про-шаговый
# ('стиль наказывает первые шаги' инвертируется). Веса 1.8 НЕ тронуты --
# одна переменная.


def _style_gate(env, command_name_vel: str = "base_velocity") -> torch.Tensor:
    cmd = env.command_manager.get_term(command_name_vel).command
    return (torch.norm(cmd[:, :3], dim=-1) >= 0.1).float()



def style_joint_pos(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cmd: StylePhaseCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    ref_jp, _ = cmd.ref_joints()
    err_sq = (((asset.data.joint_pos[:, cmd.joint_ids] - ref_jp) ** 2) * cmd.joint_pos_weights).sum(dim=-1)
    return _style_gate(env) * torch.exp(-STYLE_SCALE_JOINT_POS * err_sq)


def style_joint_vel(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cmd: StylePhaseCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    _, ref_jv = cmd.ref_joints()
    err_sq = ((asset.data.joint_vel[:, cmd.joint_ids] - ref_jv) ** 2).sum(dim=-1)
    return _style_gate(env) * torch.exp(-STYLE_SCALE_JOINT_VEL * err_sq)


def style_root_h(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    # История двух неверных форм (2026-09-03, лог TRAINING_STATE):
    # (1) root_z − env_origins.z: kernel 0.003 -- тайловые origin'ы НЕ
    #     равны высоте рельефа под роботом, вычитание добавило смещение;
    # (2) root_z − mean(calf_z): kernel 0.000 -- стоп-тел в артикуляции
    #     НЕТ (foot merged в calf, origin calf = КОЛЕНО, ~0.25 над землёй).
    # Правильная минимальная форма -- РОВНО как base_height_l2 всей
    # v2-v4-линии: СЫРАЯ мировая высота root (рельеф у нас около нуля,
    # bias на буграх -- тот же принятый bias линии, флаг №2 ревью base).
    cmd: StylePhaseCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    err = (asset.data.root_pos_w[:, 2] - cmd.ref_h()).abs()
    return _style_gate(env) * torch.exp(-STYLE_SCALE_ROOT_H * err)


def style_root_rp(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cmd: StylePhaseCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    ref_roll, ref_pitch = cmd.ref_rp()
    err = (roll - ref_roll).abs() + (pitch - ref_pitch).abs()
    return _style_gate(env) * torch.exp(-STYLE_SCALE_ROOT_RP * err)


@configclass
class UnitreeB2StyleTaskEnvCfg(UnitreeB2WalkResetEnvCfg):
    """walk_reset (v4-пакет) + style-награды slowwalk-клипа + фазовые часы."""

    def __post_init__(self):
        super().__post_init__()

        # 8192 сред -- одобрение хозяина для серьёзного IL-рана
        # (GPU-утилизация IL-тасков была 30-50% на 4096).
        self.scene.num_envs = 8192

        # ------Commands: + style_phase (velocity_commands walk_reset НЕ трогаем)------
        self.commands.style_phase = StylePhaseCommandCfg()

        # ------Observations: + (cos, sin) фазы ХВОСТОМ после стандартных 45------
        self.observations.policy.style_phase = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "style_phase"},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        self.observations.critic.style_phase = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "style_phase"},
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        # ------Rewards: style поверх task (веса ИЗМЕРЕНЫ, см. докстринг)------
        self.rewards.style_joint_pos = RewTerm(
            func=style_joint_pos, weight=STYLE_WEIGHT_JOINT_POS,
            params={"command_name": "style_phase", "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.style_joint_vel = RewTerm(
            func=style_joint_vel, weight=STYLE_WEIGHT_JOINT_VEL,
            params={"command_name": "style_phase", "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.style_root_h = RewTerm(
            func=style_root_h, weight=STYLE_WEIGHT_ROOT_H,
            params={"command_name": "style_phase", "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.style_root_rp = RewTerm(
            func=style_root_rp, weight=STYLE_WEIGHT_ROOT_RP,
            params={"command_name": "style_phase", "asset_cfg": SceneEntityCfg("robot")},
        )

        # ------Events: спавн вертикально (П2 пакета base 2026-09-03)------
        # rough_env_cfg:58-66 ПОДМЕНЯЕТ params randomize_reset_base целиком:
        # roll/pitch +-3.14 (спавн вверх ногами -- legged_gym-наследие под
        # recovery, walk_reset жил с ним ТОЛЬКО потому, что падение не
        # терминировало). С терминацией это insta-смерть + is_terminated
        # ни за что на заметной доле ресетов. Recovery -- не цель гибрида.
        # yaw/z/скорости не трогаем.
        self.events.randomize_reset_base.params["pose_range"]["roll"] = (-0.3, 0.3)
        self.events.randomize_reset_base.params["pose_range"]["pitch"] = (-0.3, 0.3)

        # ------Terminations: падение = конец эпизода (пакет base 2026-09-03)------
        # Раскладка v7-freeze: joint_pos_limits -98.8/эпизод (54% всей платы)
        # платила УПАВШАЯ туша (у walk_reset падение не терминирует, лежачий
        # хвост со смятыми в лимиты суставами живёт до конца 20с эпизода) --
        # критик честно предпочёл «лечь сразу» (-11 против -181 у ходьбы).
        # В родителе illegal_contact=None -- params не пропатчить, строим
        # DoneTerm заново (стоковая строка rough_env_cfg:153).
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces", body_names=[self.base_link_name, ".*_hip"]
                ),
                "threshold": 1.0,
            },
        )
        # Страховка от суицид-эксплойта: по-шаговый поток при плохой походке
        # отрицательный, «нырнуть и сдохнуть» обрывал бы его выгодно. Разовый
        # штраф порядка секунд лежания. Сигнатура суицида для мониторинга:
        # ep_len схлопывается при растущем reward -> штраф вверх.
        self.rewards.is_terminated.weight = -10.0

        # Зачистка нулевых термов ОБЯЗАТЕЛЬНА и именно здесь: в родителях
        # она под гвардом `__class__.__name__ == "...WalkResetEnvCfg"` --
        # для сабкласса не срабатывает, и wheel_vel_penalty (weight=0,
        # joint_names="" -- колёсные роботы) доживает до резолва и роняет
        # старт («Not all regular expressions are matched», первый запуск
        # гибрида 2026-09-03). Style-веса ненулевые -- зачистка их не
        # трогает.
        self.disable_zero_weight_rewards()
