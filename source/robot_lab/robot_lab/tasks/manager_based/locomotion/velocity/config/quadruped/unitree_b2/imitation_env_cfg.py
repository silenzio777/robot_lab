# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Путь 1 -- DeepMimic-style PMC-imitation, первый тест-клип `dog_idle_003`.

Заказ хозяина 2026-08-31: остановить WALK_RESET (аномалия value_function
loss, не разрешилась после ~100 итераций -- см. TRAINING_STATE.md), начать
imitation-обучение на ретаргетнутом mocap Лабрадора (Tencent
"lifelike-agility-and-play", ретаргет сделан агентом b2, координация
формата и дизайн-решений -- base, полный план `~/base/DISTILLED/
2026-08-31_imitation-path1-plan.md`).

Источник reward-формулы: Tencent PMC (Primitive Motor Controller),
Supplementary статьи, РЕАЛЬНЫЕ веса из их train_scripts (не дефолт кода) --
`r = 0.3*r_jp + 0.05*r_jv + 0.1*r_toe + 0.5*r_rootpos + 0.05*r_rootvel`,
каждый терм `exp(-scale*Σerr²)`.

Архитектурные решения (согласованы train+base в этом разговоре,
2026-08-31):
- phase-in-obs (нормализованное время 0->1 с начала эпизода), НЕ
  target-pose-in-obs -- DeepMimic-прецедент + generalization-аргумент
  (RAI/CMU статья в нашей базе). target-lookup всё равно нужен для
  reward-термов независимо от этого выбора -- не инженерный компромисс.
- клип НЕ цикличен -> phase без cos/sin-обёртки (та нужна только для
  зацикленных движений).
- episode заканчивается на конце клипа (episode_length_s = длина клипа),
  не зацикливается/не растягивается.
- RSI (Reference State Initialization) встроен в CommandTerm's
  _resample_command -- IsaacLab's CommandManager.reset() ВСЕГДА вызывает
  _resample_command на каждый env-reset (проверено напрямую,
  command_manager.py:120-149,172-187), независимо от resampling_time_range
  -- тот держим "практически никогда" (та же практика, что
  GaitPhaseCommandCfg), чтобы RSI срабатывал ТОЛЬКО на полный эпизод-ресет,
  не посреди эпизода.
- Порядок ресета в ManagerBasedRLEnv._reset_idx (проверено напрямую,
  manager_based_rl_env.py:351-383): event_manager.apply(mode="reset")
  ПЕРЕД command_manager.reset() -- значит наш RSI-teleport (в
  _resample_command) гарантированно выполняется ПОСЛЕДНИМ и не будет
  затёрт стоковыми reset_joints_by_scale/reset_root_state_uniform (которые
  в этом файле всё равно отключены явно, чтобы не тратить впустую работу).

Единицы/конвенции данных (`retargeted_csv/training/dog_idle_003.json`,
подтверждено чтением файла напрямую, не по памяти base):
- frame_layout = root_pos_xyz(3) + root_quat_wxyz(4) + joint_pos(12),
  120.048fps, 1545 кадров, длительность 12.8615с.
- joint_order = FR,FL,RR,RL x (hip,thigh,calf) -- БИТ-В-БИТ совпадает с
  `rough_env_cfg.py`'s собственным `joint_names` (используется уже во всех
  ObservationsCfg/ActionsCfg этого репо) -- никакой перестановки колонок
  не нужно, только явный `find_joints()`-резолв для надёжности (не
  полагаться на совпадение молча).
- root_quat в wxyz -- СОВПАДАЕТ с MuJoCo/IsaacLab's `write_root_pose_to_sim`
  собственной конвенцией (w,x,y,z) -- проверено напрямую в
  `articulation.py:355-364`, конвертация не нужна.
- foot_targets = FR,FL,RR,RL x xyz, мировые IK-цели (уже есть в JSON,
  вопреки первой версии плана base -- проверено напрямую чтением файла,
  b2 добавил их до того как понадобилось спрашивать).
- Скорости (joint_vel, root_lin/ang_vel) НЕ хранятся -- считаем конечной
  разностью по всему клипу один раз при загрузке (не на лету по кадру, но
  численно эквивалентно -- клип статичен, нет смысла пересчитывать заново
  каждый query).

НЕ включено в первый прогон (сознательно, минимизация confound'ов для
sanity-check): push_robot/external_force_torque domain randomization,
стоковые reset_joints_by_scale/reset_root_state_uniform (RSI их заменяет
полностью). Actuator-gain randomization и rigid-body-материал/масса/
инерция/CoM randomization (startup-mode) оставлены -- не конфликтуют с RSI,
общая физическая устойчивость всё ещё нужна.

Early-termination пороги -- пока ЧИСЛА TENCENT как стартовая точка (pos
err summed > 1.0 m^2, angle err > 1.0 rad), calibровка под реальные B2
ошибки первого прогона -- ОТКРЫТЫЙ пункт плана, тот же урок что
joint_acc_l2 в WALK_RESET (не переносить абсолютные числа между разными
по масштабу системами вслепую) --ПОМЕЧЕНО как непроверенное, требует
измерения на первом реальном прогоне.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg import UnitreeB2FlatEnvCfg

# ---------------------------------------------------------------------------
# Данные клипа
# ---------------------------------------------------------------------------

MOTION_JSON_PATH = Path("/home/silenzio/lib/urdf_stand/retargeted_csv/training/dog_idle_003.json")


def _read_motion_meta(path: Path) -> tuple[float, int]:
    """Дёшево читает только заголовок (frame_dt/n_frames) на этапе построения
    конфига (import time, CPU, до создания среды) -- нужно для
    episode_length_s ДО того, как CommandTerm вообще существует. Полная
    загрузка тензоров на GPU происходит отдельно, в MotionRefCommand.__init__
    (см. ниже), не здесь -- не дублировать тяжёлую загрузку дважды."""
    with open(path) as f:
        meta = json.load(f)
    frame_dt = float(meta["frame_dt"])
    n_frames = int(meta["n_frames"])
    return frame_dt, n_frames


_MOTION_FRAME_DT, _MOTION_N_FRAMES = _read_motion_meta(MOTION_JSON_PATH)
MOTION_DURATION_S = (_MOTION_N_FRAMES - 1) * _MOTION_FRAME_DT

# PMC reward -- реальные веса из train_scripts статьи (Supplementary Tencent
# lifelike-agility-and-play), не дефолт кода. Полное обоснование --
# `~/base/DISTILLED/2026-08-31_imitation-path1-plan.md`.
PMC_WEIGHT_JOINT_POS = 0.3
PMC_WEIGHT_JOINT_VEL = 0.05
PMC_WEIGHT_TOE = 0.1
PMC_WEIGHT_ROOT_POSE = 0.5
PMC_WEIGHT_ROOT_VEL = 0.05

PMC_SCALE_JOINT_POS = 1.0
PMC_SCALE_JOINT_VEL = 0.1
PMC_SCALE_TOE = 40.0
PMC_SCALE_ROOT_POS = 20.0
PMC_SCALE_ROOT_ANGLE = 10.0
PMC_SCALE_ROOT_LIN_VEL = 2.0
PMC_SCALE_ROOT_ANG_VEL = 0.2

# Early-termination -- Tencent-числа (1.0/1.0) были непроверенными под B2 --
# перекалиброваны 2026-08-31 по факту (train: RSI-error-distribution замер
# на b2_imitation_4999_final, 30 стартов, тот же метод, что termination
# использует): p99 ошибки ВНУТРИ здорового (до-расхождения) участка уже
# 0.93-0.96 у обеих величин -- 1.0 был впритык к естественному потолку
# здорового трекинга, слишком туго. Умеренное расширение (base: "~1.0->1.3",
# дёшево, применяем в любом случае) -- даёт немного пространства
# recoverable-заминкам (было 18% случаев restore within window), не
# открывая шлюз для катастрофических срывов (те растут в разы за порог
# 10-35x, никакой разумный подъём порога их не поймает). См.
# train_research/TRAINING_STATE.md записи ~08:4x/~12:2x для полных цифр.
DIVERGE_POS_ERR_SQ_THRESHOLD = 1.3  # m^2, суммарно по root+joints (см. termination-функцию)
DIVERGE_ANGLE_ERR_THRESHOLD = 1.3  # rad

# Prioritized RSI по бинам фазы (2026-09-01, train+base) -- найдено:
# честный per-phase sweep на реальном чекпоинте (it3100 первого from-scratch
# на патченном референсе) с ЧЕСТНЫМ RSI-телепортом показал провал (>90°
# наклон) на 17/20 точек РАВНОМЕРНО ПО ВСЕМУ КЛИПУ, не только у self-collision
# сегмента -- обычный uniform-RSI тратит одинаковый бюджет семплов на уже
# решённые и на упорно нерешаемые участки. Идея Tencent (per-clip
# P(clip)~(1-avg_reward)^3, применяем ту же логику на уровне бинов ВНУТРИ
# одного клипа, т.к. у нас всего 1 клип, а не пачка): трекаем EMA "какую долю
# ОСТАВШЕГОСЯ клипа обычно проходят, стартуя из этого бина" -- сигнал уже
# бесплатно доступен в _resample_command (self.ref_time в момент вызова, ДО
# перезаписи, = докуда доехал эпизод, стартовавший в предыдущем бине), не
# нужен отдельный reward-хук. Сэмплируем следующий старт с весом
# (1-прогресс)^PHASE_PRIORITY_POWER, тяжёлые бины оверсэмплятся автоматически
# и динамически (не статичный снимок с одного чекпоинта -- веса живут и
# двигаются вместе с тем, что политика реально освоила).
#
# PHASE_PRIORITY_FLOOR (base's находка, PER-ловушка): чистая (1-progress)^p
# без пола может выжать вероятность у УЖЕ решённых бинов почти до нуля -- то
# самое "чиним одно, роняем другое" prioritized-replay забывание. floor=0.15
# гарантирует, что даже полностью решённый бин (progress=1) сохраняет 15%
# от максимального веса -- достаточно, чтобы политика продолжала изредка
# видеть эти состояния и не забывала их, но всё ещё сильно смещает бюджет
# семплов к трудным местам (0.15 vs 1.0 -- почти 7-кратный перекос).
N_PHASE_BINS = 16
PHASE_PRIORITY_POWER = 2.0
# 0.15 -> 1.0 (2026-09-01, диагностический эксперимент train+base): три
# попытки prioritized-RSI (баг/фикс) от it8800 обе дрейфовали ХУЖЕ
# (14->16->17 FAIL), непонятно -- это сама приоритизация нестабильна на
# этой задаче, или обычный дрейф/оверфит на одном клипе независимо от
# схемы сэмплинга. floor=1.0 делает weight=floor+(1-floor)*(...)=1.0 для
# ВСЕХ бинов равномерно -- математически идентично чистому uniform-RSI
# (то же самое, что было до всей этой priority-затеи, it3100-схема), но
# через уже существующий параметр, без переписывания алгоритма (меньше
# риск внести новый баг в контрольный прогон). Изолирует переменную:
# если uniform-resume от it8800 ТОЖЕ дрейфует хуже -- дело не в
# приоритизации. Временное значение для контрольного эксперимента, не
# постоянное -- см. train_research/TRAINING_STATE.md за исходом.
PHASE_PRIORITY_FLOOR = 1.0
PHASE_PRIORITY_EMA_ALPHA = 0.02  # медленная EMA -- 4096 envs/итерация, апдейтов много

# base's находка 2026-08-31 (проверено b2 на JSON напрямую): foot_targets --
# ЦЕНТРЫ СФЕР стопы (радиус 0.032м), не точки контакта с полом. Merge_fixed_
# joints=True при импорте URDF схлопывает "*_foot" (сфера) в "*_calf" (нет
# отдельного foot-тела в резолвленном списке тел -- см. rough_env_cfg.py's
# собственный комментарий), значит body_pos_w("*_calf") -- это ФРЕЙМ ТЕЛА
# calf, НЕ сфера стопы. Офсет сферы относительно calf -- прочитан напрямую
# из URDF (*_foot_joint's origin xyz, идентично для FL/FR/RL, RR отличается
# на ~0.087мм по Y, игнорируем как шум): (0, 0, -0.35)м вдоль локальной оси
# calf. Без этой поправки -- систематическая ошибка ~0.35м (не 0.032м, тот
# только для z-высоты в момент контакта) на КАЖДОМ кадре toe-терма.
FOOT_SPHERE_LOCAL_OFFSET = (0.0, 0.0, -0.35)


# ---------------------------------------------------------------------------
# Батч-совместимый slerp -- IsaacLab's quat_slerp (math.py:1730) ЯВНО не
# батчится ("This function does not support batch processing") -- нужен свой
# для тысяч параллельных сред с разной per-env фазой.
# ---------------------------------------------------------------------------


def _batched_quat_slerp(q1: torch.Tensor, q2: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """q1, q2: (..., 4) wxyz. tau: (...,) в [0,1]. Стандартная batched-slerp
    формула, с nlerp-фолбэком при почти нулевом угле (избегает деления на
    ~0 в sin(angle))."""
    dot = (q1 * q2).sum(-1, keepdim=True)
    q2 = torch.where(dot < 0, -q2, q2)
    dot = torch.where(dot < 0, -dot, dot).clamp(-1.0, 1.0)
    angle = torch.acos(dot)
    sin_angle = torch.sin(angle)
    tau_ = tau.unsqueeze(-1)
    nlerp = torch.nn.functional.normalize((1.0 - tau_) * q1 + tau_ * q2, dim=-1)
    safe_sin = torch.where(sin_angle.abs() < 1e-6, torch.ones_like(sin_angle), sin_angle)
    w1 = torch.sin((1.0 - tau_) * angle) / safe_sin
    w2 = torch.sin(tau_ * angle) / safe_sin
    slerp = w1 * q1 + w2 * q2
    return torch.where(angle.abs() < 1e-4, nlerp, torch.nn.functional.normalize(slerp, dim=-1))


# ---------------------------------------------------------------------------
# MotionRefCommand -- motion-buffer + per-env reference-time state + RSI.
# Тот же CommandTerm-паттерн, что GaitPhaseCommand (walk_env_cfg.py) --
# единственный прецедент в этом репо для "per-env непрерывное состояние,
# продвигаемое каждый шаг" -- переиспользован, не изобретён с нуля. НЕТ
# прецедента в ManagerBasedRLEnv для загрузки внешнего файла в GPU-тензор
# при построении среды (только в Direct-workflow g1_amp/motion_loader.py) --
# эта часть, наоборот, новый код, пришлось спроектировать самому.
# ---------------------------------------------------------------------------


class MotionRefCommand(CommandTerm):
    """Владеет: (1) референс-траекторией на GPU (позиции+интерполированные
    по конечной разности скорости), (2) per-env `ref_time` (секунды с начала
    ТЕКУЩЕГО эпизода в системе координат клипа), (3) RSI -- teleport робота
    на случайный момент клипа при каждом env-reset. `command` (то, что видит
    политика через ObsTerm) -- ТОЛЬКО нормализованная фаза (0->1), скаляр,
    без sin/cos-обёртки (клип не цикличен). Reward/termination-функции ниже
    читают куда более богатое состояние (`query()`) напрямую через
    `env.command_manager.get_term("motion_ref")`, не через `command`."""

    cfg: "MotionRefCommandCfg"

    def __init__(self, cfg: "MotionRefCommandCfg", env) -> None:
        super().__init__(cfg, env)
        self.asset: Articulation = env.scene[cfg.asset_name]

        with open(cfg.motion_json_path) as f:
            data = json.load(f)
        assert data["frame_layout"] == "root_pos_xyz(3) + root_quat_wxyz(4) + joint_pos(12)", (
            f"Неожиданный frame_layout в {cfg.motion_json_path}: {data['frame_layout']!r} -- "
            "код ниже жёстко предполагает этот layout, не парсит его динамически."
        )
        assert data["root_quat_convention"] == "wxyz"

        frames = torch.tensor(data["frames"], dtype=torch.float32, device=self.device)  # (F, 19)
        foot_targets_flat = torch.tensor(data["foot_targets"], dtype=torch.float32, device=self.device)  # (F, 12)
        self.n_frames = frames.shape[0]
        self.frame_dt = float(data["frame_dt"])
        self.duration_s = (self.n_frames - 1) * self.frame_dt

        self.ref_root_pos = frames[:, 0:3]
        self.ref_root_quat = frames[:, 3:7]
        self.ref_joint_pos_raw = frames[:, 7:19]  # порядок = data["joint_order"], НЕ обязательно порядок модели
        self.ref_foot_targets = foot_targets_flat.view(self.n_frames, 4, 3)  # FR,FL,RR,RL x xyz

        # Резолв joint_ids модели В ПОРЯДКЕ данных (не полагаемся на то, что
        # порядок модели молча совпадает, хотя по rough_env_cfg.py и должен) --
        # self.joint_ids используется КАЖДЫЙ раз при чтении/записи joint-состояния,
        # гарантируя согласованность колонок с ref_joint_pos_raw независимо от
        # внутреннего порядка Articulation.
        joint_ids, found_names = self.asset.find_joints(data["joint_order"], preserve_order=True)
        assert found_names == data["joint_order"], (found_names, data["joint_order"])
        self.joint_ids = joint_ids

        # Тот же резолв для стоп (foot_targets -- FR,FL,RR,RL) -- calf-тело,
        # тот же foot-proxy convention, что весь остальной репо (foot_link_name).
        foot_body_names = ["FR_calf", "FL_calf", "RR_calf", "RL_calf"]
        body_ids, found_body_names = self.asset.find_bodies(foot_body_names, preserve_order=True)
        assert found_body_names == foot_body_names
        self.foot_body_ids = body_ids

        # Конечно-разностные скорости -- предвычислены один раз по всему
        # клипу (клип статичен, пересчитывать на каждый query бессмысленно).
        # Центральная разность внутри, вперёд/назад на краях.
        self.ref_joint_vel_raw = self._finite_diff(self.ref_joint_pos_raw, self.frame_dt)
        self.ref_root_lin_vel = self._finite_diff(self.ref_root_pos, self.frame_dt)
        self.ref_root_ang_vel = self._finite_diff_angular(self.ref_root_quat, self.frame_dt)

        self.ref_time = torch.zeros(self.num_envs, device=self.device)

        # Prioritized RSI по бинам фазы -- см. константы N_PHASE_BINS и
        # соседей выше за полным обоснованием. bin_difficulty стартует
        # РОВНО в 1.0 (максимальный вес = чистый uniform на первом же
        # ресете, пока EMA ещё не накопила ни одного реального замера) --
        # веса расходятся от uniform ТОЛЬКО по мере поступления данных.
        self.bin_width = self.duration_s / N_PHASE_BINS
        self.bin_difficulty = torch.ones(N_PHASE_BINS, device=self.device)
        # -1 = "эпизода ещё не было, нечего засчитывать в EMA" (первый вызов
        # _resample_command на каждый env, до перезаписи -- отличаем от
        # реального бина 0 явным сентинелом, не нулём).
        self.start_bin = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        # ТОЧНОЕ время старта (не bin_idx*bin_width -- та аппроксимация
        # игнорирует джиттер внутри бина, до +-bin_width систематическая
        # ошибка в progress, нашли живьём 2026-09-01: 2800 итераций
        # приоритизированного сэмплинга с этим багом дали РЕГРЕССИЮ
        # 14/20->18-19/20 FAIL на честном phase-sweep -- завышенный
        # progress занижал difficulty именно средне-трудных бинов,
        # сэмплинг съезжал не туда. self.start_bin остаётся (нужен для
        # индекса EMA-бина), start_time хранится отдельно для точной
        # арифметики progress.
        self.start_time = torch.zeros(self.num_envs, device=self.device)

    @staticmethod
    def _finite_diff(x: torch.Tensor, dt: float) -> torch.Tensor:
        v = torch.zeros_like(x)
        v[1:-1] = (x[2:] - x[:-2]) / (2.0 * dt)
        v[0] = (x[1] - x[0]) / dt
        v[-1] = (x[-1] - x[-2]) / dt
        return v

    @staticmethod
    def _finite_diff_angular(q: torch.Tensor, dt: float) -> torch.Tensor:
        # quat_box_minus(q2,q1) = log(q2 * q1^-1) -- мировой угол-ось вектор,
        # делённый на dt даёт угловую скорость (math.py:586-600, проверено напрямую).
        w = torch.zeros(q.shape[0], 3, device=q.device, dtype=q.dtype)
        w[1:-1] = math_utils.quat_box_minus(q[2:], q[:-2]) / (2.0 * dt)
        w[0] = math_utils.quat_box_minus(q[1], q[0]) / dt
        w[-1] = math_utils.quat_box_minus(q[-1], q[-2]) / dt
        return w

    def query(self, times: torch.Tensor) -> dict[str, torch.Tensor]:
        """times: (N,) секунды, клампится в [0, duration_s] -- запрос ЗА
        пределами клипа замораживается на последнем кадре (не оборачивается,
        не экстраполируется -- см. модульный докстринг про "эпизод
        заканчивается на конце клипа")."""
        t = times.clamp(0.0, self.duration_s)
        frame_f = t / self.frame_dt
        idx0 = frame_f.long().clamp(0, self.n_frames - 2)
        idx1 = idx0 + 1
        alpha = (frame_f - idx0.to(frame_f.dtype)).clamp(0.0, 1.0)
        a1 = alpha.unsqueeze(-1)

        root_pos = torch.lerp(self.ref_root_pos[idx0], self.ref_root_pos[idx1], a1)
        root_quat = _batched_quat_slerp(self.ref_root_quat[idx0], self.ref_root_quat[idx1], alpha)
        joint_pos = torch.lerp(self.ref_joint_pos_raw[idx0], self.ref_joint_pos_raw[idx1], a1)
        joint_vel = torch.lerp(self.ref_joint_vel_raw[idx0], self.ref_joint_vel_raw[idx1], a1)
        root_lin_vel = torch.lerp(self.ref_root_lin_vel[idx0], self.ref_root_lin_vel[idx1], a1)
        root_ang_vel = torch.lerp(self.ref_root_ang_vel[idx0], self.ref_root_ang_vel[idx1], a1)
        foot_targets = torch.lerp(self.ref_foot_targets[idx0], self.ref_foot_targets[idx1], a1.unsqueeze(-1))
        return {
            "root_pos": root_pos,
            "root_quat": root_quat,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "root_lin_vel": root_lin_vel,
            "root_ang_vel": root_ang_vel,
            "foot_targets": foot_targets,
        }

    @property
    def command(self) -> torch.Tensor:
        # Политика видит ТОЛЬКО фазу (0->1), не target-pose -- архитектурное
        # решение, см. модульный докстринг. Не цикличен -> без sin/cos.
        return (self.ref_time / self.duration_s).clamp(0.0, 1.0).unsqueeze(-1)

    def _update_metrics(self):
        self.metrics["phase_mean"] = self.ref_time / self.duration_s

    def _resample_command(self, env_ids):
        """RSI: старт-момент клипа + телепорт робота на этот референс-кадр.
        Вызывается CommandManager.reset() на КАЖДОМ env-reset безусловно
        (проверено напрямую, см. модульный докстринг) -- resampling_time_range
        держим огромным, чтобы это НЕ срабатывало дополнительно посреди
        эпизода (см. Cfg ниже).

        2026-09-01: старт-момент теперь НЕ чистый uniform -- prioritized
        по бинам фазы, см. константы N_PHASE_BINS/PHASE_PRIORITY_* выше за
        полным обоснованием (честный per-phase sweep нашёл провал почти
        по всему клипу, не только в одном месте -- обычный uniform-RSI
        тратит одинаковый бюджет на решённое и на нерешаемое)."""
        n = len(env_ids)
        if n == 0:
            return
        env_ids_t = torch.as_tensor(env_ids, device=self.device)

        # --- EMA-апдейт по СТАРОМУ бину (докуда доехал эпизод, который сейчас
        # завершается) -- self.ref_time тут ЕЩЁ старое значение (перезапишем
        # ниже), это и есть "докуда дошли до срабатывания termination/timeout".
        had_prior_episode = self.start_bin[env_ids_t] >= 0
        if had_prior_episode.any():
            done_idx = env_ids_t[had_prior_episode]
            prior_bin = self.start_bin[done_idx]
            prior_start_time = self.start_time[done_idx]  # ТОЧНОЕ время (с джиттером), не bin_idx*bin_width
            achieved = self.ref_time[done_idx]
            remaining = (self.duration_s - prior_start_time).clamp(min=1e-6)
            progress = ((achieved - prior_start_time) / remaining).clamp(0.0, 1.0)
            # index_add_ + отдельный счётчик вместо scatter-mean -- несколько
            # envs МОГУТ завершиться в один и тот же бин на одном ресете
            # (4096 параллельных сред), обычное присваивание потеряло бы все,
            # кроме последнего; так усредняем честно по факту, затем EMA.
            bin_sum = torch.zeros(N_PHASE_BINS, device=self.device)
            bin_count = torch.zeros(N_PHASE_BINS, device=self.device)
            bin_sum.index_add_(0, prior_bin, progress)
            bin_count.index_add_(0, prior_bin, torch.ones_like(progress))
            touched = bin_count > 0
            batch_mean_progress = torch.zeros(N_PHASE_BINS, device=self.device)
            batch_mean_progress[touched] = bin_sum[touched] / bin_count[touched]
            batch_difficulty = 1.0 - batch_mean_progress
            self.bin_difficulty[touched] = (
                (1.0 - PHASE_PRIORITY_EMA_ALPHA) * self.bin_difficulty[touched]
                + PHASE_PRIORITY_EMA_ALPHA * batch_difficulty[touched]
            )

        # --- Новый старт: сэмплируем бин пропорционально приоритету, джиттер
        # внутри бина непрерывный (не залипаем на границах бинов).
        weight = PHASE_PRIORITY_FLOOR + (1.0 - PHASE_PRIORITY_FLOOR) * self.bin_difficulty.clamp(
            0.0, 1.0
        ) ** PHASE_PRIORITY_POWER
        new_bin = torch.multinomial(weight, n, replacement=True)
        jitter = torch.rand(n, device=self.device)
        start_times = (new_bin.to(self.ref_time.dtype) + jitter) * self.bin_width
        self.start_bin[env_ids_t] = new_bin
        self.start_time[env_ids_t] = start_times
        self.ref_time[env_ids_t] = start_times

        ref = self.query(start_times)

        # root pose: позиция + env_origins (мировая точка отсчёта конкретной
        # параллельной среды -- см. events.py:896,948,1161 за тем же паттерном),
        # кватернион wxyz напрямую (та же конвенция, что write_root_pose_to_sim).
        root_pos_world = ref["root_pos"] + self._env.scene.env_origins[env_ids_t]
        root_pose = torch.cat([root_pos_world, ref["root_quat"]], dim=-1)
        self.asset.write_root_pose_to_sim(root_pose, env_ids=env_ids_t)

        # Референс-скорости корня в СВОЁМ (клипа) мировом фрейме -- тот же
        # фрейм, что root_pos (не root-локальный) -- write_root_velocity_to_sim
        # ожидает МИРОВУЮ линейную/угловую скорость (см. articulation.py:441,
        # аналогичная конвенция write_root_pose_to_sim).
        root_vel = torch.cat([ref["root_lin_vel"], ref["root_ang_vel"]], dim=-1)
        self.asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids_t)

        self.asset.write_joint_state_to_sim(
            ref["joint_pos"], ref["joint_vel"], joint_ids=self.joint_ids, env_ids=env_ids_t
        )

    def _update_command(self):
        # Монотонно продвигаем время для ВСЕХ сред каждый шаг (не гейтится
        # ни на что -- imitation всегда "играет", в отличие от
        # GaitPhaseCommand's should_move-заморозки, той семантики тут нет).
        self.ref_time = (self.ref_time + self._env.step_dt).clamp(0.0, self.duration_s)


@configclass
class MotionRefCommandCfg(CommandTermCfg):
    class_type: type = MotionRefCommand
    motion_json_path: Path = MOTION_JSON_PATH
    # "Практически никогда" -- тот же паттерн, что GaitPhaseCommandCfg
    # (walk_env_cfg.py:216) -- периодический ре-семпл посреди эпизода НЕ
    # нужен для первого теста (RSI только на полный эпизод-ресет).
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)
    asset_name: str = "robot"
    debug_vis: bool = False


# ---------------------------------------------------------------------------
# PMC reward-термы -- читают MotionRefCommand напрямую через
# env.command_manager.get_term(command_name), не через .command (та отдаёт
# только фазу политике, не полное референс-состояние).
# ---------------------------------------------------------------------------


def imitation_joint_pos(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cmd: MotionRefCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    ref = cmd.query(cmd.ref_time)
    sim_joint_pos = asset.data.joint_pos[:, cmd.joint_ids]
    err_sq = ((sim_joint_pos - ref["joint_pos"]) ** 2).sum(dim=-1)
    return torch.exp(-PMC_SCALE_JOINT_POS * err_sq)


def imitation_joint_vel(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cmd: MotionRefCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    ref = cmd.query(cmd.ref_time)
    sim_joint_vel = asset.data.joint_vel[:, cmd.joint_ids]
    err_sq = ((sim_joint_vel - ref["joint_vel"]) ** 2).sum(dim=-1)
    return torch.exp(-PMC_SCALE_JOINT_VEL * err_sq)


def _sim_foot_sphere_pos_w(asset: Articulation, foot_body_ids: list[int]) -> torch.Tensor:
    """Мировая позиция ЦЕНТРА СФЕРЫ стопы (не тела calf) -- см.
    FOOT_SPHERE_LOCAL_OFFSET's докстринг выше за обоснованием. Тот же
    quat_apply-паттерн, что весь остальной репо использует для body-local ->
    world трансформаций (math.py:625-643)."""
    calf_pos_w = asset.data.body_pos_w[:, foot_body_ids]  # (N,4,3)
    calf_quat_w = asset.data.body_quat_w[:, foot_body_ids]  # (N,4,4) wxyz
    offset = torch.tensor(FOOT_SPHERE_LOCAL_OFFSET, device=asset.data.body_pos_w.device).expand(
        calf_pos_w.shape[0], calf_pos_w.shape[1], 3
    )
    return calf_pos_w + math_utils.quat_apply(calf_quat_w, offset)


def imitation_toe(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Toe-позиция ОТНОСИТЕЛЬНО корня (не мировая) -- та же практика, что
    оригинальный DeepMimic's end-effector reward (форма стойки/шага, не
    абсолютное положение в комнате -- иначе робот наказывался бы за
    легитимный снос старта, никак не связанный с качеством имитации)."""
    cmd: MotionRefCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    ref = cmd.query(cmd.ref_time)

    sim_root_pos = asset.data.root_pos_w  # (N,3), мировая
    sim_foot_pos = _sim_foot_sphere_pos_w(asset, cmd.foot_body_ids)  # (N,4,3), центр сферы, мировая
    sim_foot_rel = sim_foot_pos - sim_root_pos.unsqueeze(1)

    ref_foot_rel = ref["foot_targets"] - ref["root_pos"].unsqueeze(1)

    err_sq = ((sim_foot_rel - ref_foot_rel) ** 2).sum(dim=(-1, -2))
    return torch.exp(-PMC_SCALE_TOE * err_sq)


def imitation_root_pose(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Совмещённый позиция+ориентация терм (PMC's r_rootpos, вес 0.5,
    scale=[-20 pos,-10 angle] -- одна exp с двумя слагаемыми в показателе,
    математически = произведению двух отдельных exp)."""
    cmd: MotionRefCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    ref = cmd.query(cmd.ref_time)

    sim_root_pos_local = asset.data.root_pos_w - env.scene.env_origins
    pos_err_sq = ((sim_root_pos_local - ref["root_pos"]) ** 2).sum(dim=-1)

    angle_err = math_utils.quat_error_magnitude(asset.data.root_quat_w, ref["root_quat"])
    angle_err_sq = angle_err**2

    return torch.exp(-(PMC_SCALE_ROOT_POS * pos_err_sq + PMC_SCALE_ROOT_ANGLE * angle_err_sq))


def imitation_root_vel(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cmd: MotionRefCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    ref = cmd.query(cmd.ref_time)

    lin_err_sq = ((asset.data.root_lin_vel_w - ref["root_lin_vel"]) ** 2).sum(dim=-1)
    ang_err_sq = ((asset.data.root_ang_vel_w - ref["root_ang_vel"]) ** 2).sum(dim=-1)

    return torch.exp(-(PMC_SCALE_ROOT_LIN_VEL * lin_err_sq + PMC_SCALE_ROOT_ANG_VEL * ang_err_sq))


# ---------------------------------------------------------------------------
# Early-termination по расхождению -- числа Tencent, непроверенные под B2
# (см. модульный докстринг).
# ---------------------------------------------------------------------------


def imitation_diverged(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    pos_err_sq_threshold: float,
    angle_err_threshold: float,
) -> torch.Tensor:
    cmd: MotionRefCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    ref = cmd.query(cmd.ref_time)

    sim_root_pos_local = asset.data.root_pos_w - env.scene.env_origins
    pos_err_sq = ((sim_root_pos_local - ref["root_pos"]) ** 2).sum(dim=-1)
    angle_err = math_utils.quat_error_magnitude(asset.data.root_quat_w, ref["root_quat"])

    return (pos_err_sq > pos_err_sq_threshold) | (angle_err > angle_err_threshold)


# ---------------------------------------------------------------------------
# Env cfg
# ---------------------------------------------------------------------------


@configclass
class UnitreeB2ImitationEnvCfg(UnitreeB2FlatEnvCfg):
    """PMC-imitation поверх `UnitreeB2FlatEnvCfg` (плоский пол, без
    height_scan/terrain-curriculum -- то же обоснование, что WALK_RESET's
    отказ от кастомной машинерии, но для imitation рельеф вообще не нужен
    для первого теста одного короткого клипа)."""

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Episode length = длина клипа------------------------------
        self.episode_length_s = MOTION_DURATION_S

        # ------------------------------Commands: motion_ref заменяет base_velocity------------------------------
        self.commands.base_velocity = None
        self.commands.motion_ref = MotionRefCommandCfg()

        # ------------------------------Observations: phase вместо velocity_commands------------------------------
        # base_lin_vel уже None в RoughEnvCfg (policy), не трогаем -- та же
        # sim-to-real асимметрия actor/critic, не специфично для imitation.
        self.observations.policy.velocity_commands = None
        self.observations.critic.velocity_commands = None
        self.observations.policy.ref_phase = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "motion_ref"},
            clip=(0.0, 1.0),
            scale=1.0,
        )
        self.observations.critic.ref_phase = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "motion_ref"},
            clip=(0.0, 1.0),
            scale=1.0,
        )

        # ------------------------------Rewards: выключаем ВСЮ velocity-tracking/gait экономику------------------------------
        # PMC reward заменяет её целиком -- не "довесок" поверх локомоционных
        # термов, отдельная задача (имитация, не velocity-tracking).
        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_gait.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.upward.weight = 0.0
        self.rewards.stand_still_without_cmd.weight = 0.0
        self.rewards.joint_pos_penalty.weight = 0.0
        # Найдено живым прогоном (не заранее) -- feet_contact_without_cmd
        # читает env.command_manager.get_command("base_velocity"), которого
        # больше нет (self.commands.base_velocity = None выше) -- KeyError
        # на первом же step(). rough_env_cfg.py включает его (0.1), не
        # выключен по умолчанию как большинство cmd-based термов.
        self.rewards.feet_contact_without_cmd.weight = 0.0
        # Базовые физические регуляторы (torque/power/action_rate/contact) --
        # оставляем как есть из UnitreeB2RoughEnvCfg, они не про gait-экономику,
        # общая физическая разумность нужна и для imitation.

        # ------------------------------Rewards: 5 PMC-термов------------------------------
        self.rewards.imitation_joint_pos = RewTerm(
            func=imitation_joint_pos,
            weight=PMC_WEIGHT_JOINT_POS,
            params={"command_name": "motion_ref", "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.imitation_joint_vel = RewTerm(
            func=imitation_joint_vel,
            weight=PMC_WEIGHT_JOINT_VEL,
            params={"command_name": "motion_ref", "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.imitation_toe = RewTerm(
            func=imitation_toe,
            weight=PMC_WEIGHT_TOE,
            params={"command_name": "motion_ref", "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.imitation_root_pose = RewTerm(
            func=imitation_root_pose,
            weight=PMC_WEIGHT_ROOT_POSE,
            params={"command_name": "motion_ref", "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.imitation_root_vel = RewTerm(
            func=imitation_root_vel,
            weight=PMC_WEIGHT_ROOT_VEL,
            params={"command_name": "motion_ref", "asset_cfg": SceneEntityCfg("robot")},
        )

        # ------------------------------Events: RSI заменяет стоковые reset_joints/reset_base------------------------------
        self.events.randomize_reset_joints = None
        self.events.randomize_reset_base = None
        # Push/external-force domain randomization выключены для первого
        # sanity-check прогона (минимизация confound'ов) -- см. модульный докстринг.
        self.events.randomize_push_robot = None
        self.events.randomize_apply_external_force_torque = None

        # ------------------------------Terminations: early-stop по расхождению------------------------------
        self.terminations.imitation_diverged = DoneTerm(
            func=imitation_diverged,
            params={
                "command_name": "motion_ref",
                "asset_cfg": SceneEntityCfg("robot"),
                "pos_err_sq_threshold": DIVERGE_POS_ERR_SQ_THRESHOLD,
                "angle_err_threshold": DIVERGE_ANGLE_ERR_THRESHOLD,
            },
        )

        # ------------------------------Curriculum: не про imitation, выключено------------------------------
        self.curriculum.command_levels = None

        if self.__class__.__name__ == "UnitreeB2ImitationEnvCfg":
            self.disable_zero_weight_rewards()
