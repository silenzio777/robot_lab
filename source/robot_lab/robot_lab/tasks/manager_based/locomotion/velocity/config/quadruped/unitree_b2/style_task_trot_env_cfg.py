"""Style+Task гибрид, ШАГ 2: рысь как ВТОРОЙ style-термин поверх рабочей
базы v14 (design train+base, 2026-09-04, после подтверждённого хозяином
медленного гибрида -- confirmed_skills/hybrid_style_walk_v14/).

Ключевые решения (design-сессия, зафиксированы с base):
- ДВА ПАРАЛЛЕЛЬНЫХ StylePhaseCommand-инстанса, ОБА работают ПОСТОЯННО
  (не переключаются) -- ни один трекер никогда не "холодный" в момент
  смены активного стиля, разрыва структурно нет.
- REWARD-SELECTION (не pose-blend) по величине команды: hard threshold
  |cmd|>=0.7 включает style_trot термы, style_slow_walk термы остаются
  с ПРЕЖНИМ гейтом |cmd|>=0.1 (не тронут) -- в зоне [0.7,1.0] оба стиля
  активны одновременно, это сознательно оставлено как измеримая точка
  первого прогона, не решено заранее.
- Только SLOW-WALK владеет RSI-телепортом на ресете (enable_rsi_teleport
  True/False) -- два независимых телепорта конфликтовали бы (второй
  перезаписывает asset поверх первого, "фаза и поза -- одна переменная"
  ломается для проигравшего). Trot-трекер стартует matched_time с
  тем же случайным draw, что и slow-walk (тот же принцип "не выдумывать
  вторую эвристику"), сам ничего не пишет в sim -- окно само находит
  локально лучший матч на первых шагах, независимо от того, кто
  физически поставил стартовую позу.
- window_tempo_scale развязывает темп windowed-поиска от записанного
  frame_dt клипа: rescale15-клип рыси "подразумевает" 1.494 м/с
  (замерено напрямую по root-смещению за цикл), а наш реальный потолок
  команд ~1.0 м/с -- без развязки monotonic-clamp хронически упирался
  бы в нижний край КАЖДЫЙ шаг (не изредка как задуман джиттер), тот же
  класс проблемы, что топтание внутри окна у v12. scale = 1.0/1.494.
  Форма позы (диагональная фазировка, foot-placement) НЕ искажается --
  трогаем только скорость поиска по клипу, не референсные суставы.
- Веса per-joint для style_trot: СТАРТ UNIFORM (1.0 везде), НЕ
  переиспользуем thigh x2 от slow-walk (другой клип, другая физика).
  ПРЕДСКАЗАНИЕ зарегистрировано ДО первого замера (не задним числом):
  var_hip/thigh/calf клипа rescale15 = 0.00686/0.09613/0.05176 rad²
  -> var_hip:calf=7.55x, var_thigh:calf=0.54x -- thigh в trot-клипе
  ОТНОСИТЕЛЬНО СЛАБЕЕ calf (наоборот к slow-walk, где thigh был сжат
  относительно calf). Если короткое диагностическое окно (~1000-1500
  ит) покажет именно thigh НЕДОкорректированным -- предсказание
  подтверждено, веса ∝1/var (thigh~1.85x, hip~7.55x) применяются
  СРАЗУ следующим шагом без повторного угадывания. Slow-walk клип у
  ПРЕДЫДУЩЕГО фикса ПОСЛЕ фазовой блокировки уже не переносится 1:1 --
  var trot-клипа полезный контекст, не гарантия (фазовый механизм
  теперь работает с самого старта trot-обучения, другая ситуация).
- Клип rescale15 -- признанная физическая несамосогласованность
  (суставы движутся как для исходных 2.6 м/с, просто растянуты по
  времени до 1.5 м/с, позы НЕ пересчитаны -- заметка в самом JSON).
  Это ПРИЧИНА, почему не гонимся за точным темпом (window_tempo_scale
  решает лишь скорость ПОИСКА, не чинит саму несамосогласованность
  формы) -- используем клип как СТИЛЕВУЮ ПОДСКАЗКУ по форме ноги, не
  как точный временной эталон. Оправдано и тем, что диагональная
  координация ног УЖЕ эмерджентно возникает у v14 на vx=1.0 (78-91%
  синхронность) БЕЗ trot-референса вообще -- достаточно подсказки, не
  нужен точный temporal match.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .style_task_env_cfg import (
    StylePhaseCommand,
    StylePhaseCommandCfg,
    UnitreeB2StyleTaskEnvCfg,
    _style_gate,
)

STYLE_TROT_MOTION_JSON = Path(
    "/home/silenzio/lib/urdf_stand/retargeted_csv/training/dog_idle_003_walk_clip_wide10cm_rescale15.json"
)

# Клип-собственная скорость (замерено напрямую по root-смещению за цикл,
# 2026-09-04): 1.494 м/с. Наш текущий потолок команд ~1.0 м/с.
STYLE_TROT_CLIP_NATIVE_SPEED = 1.494
STYLE_TROT_TARGET_OPERATING_SPEED = 1.0
STYLE_TROT_WINDOW_TEMPO_SCALE = STYLE_TROT_TARGET_OPERATING_SPEED / STYLE_TROT_CLIP_NATIVE_SPEED

# Reward-selection порог: style_trot термы активны только при |cmd|>=0.7
# (zона высокой скорости, где рысь физически осмысленна); style_slow_walk
# термы остаются с прежним гейтом >=0.1 -- НЕ тронуты этим шагом.
STYLE_TROT_GATE_MIN_CMD = 0.7

# --- шкалы -- те же порядки величины, что у slow-walk (design intent:
# style-сумма ~40% от task-суммы, тот же коридор, см. докстринг
# style_task_env_cfg.py) -- ПЕРВЫЙ прогон, не финальная калибровка ---
STYLE_TROT_SCALE_JOINT_POS = 1.0
STYLE_TROT_SCALE_JOINT_VEL = 0.01
STYLE_TROT_SCALE_ROOT_H = 20.0
STYLE_TROT_SCALE_ROOT_RP = 10.0
STYLE_TROT_WEIGHT_JOINT_POS = 1.0
STYLE_TROT_WEIGHT_JOINT_VEL = 0.15
STYLE_TROT_WEIGHT_ROOT_H = 0.3
STYLE_TROT_WEIGHT_ROOT_RP = 0.35


def style_trot_joint_pos(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cmd: StylePhaseCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    ref_jp, _ = cmd.ref_joints()
    err_sq = (((asset.data.joint_pos[:, cmd.joint_ids] - ref_jp) ** 2) * cmd.joint_pos_weights).sum(dim=-1)
    return _style_gate(env, min_cmd=STYLE_TROT_GATE_MIN_CMD) * torch.exp(-STYLE_TROT_SCALE_JOINT_POS * err_sq)


def style_trot_joint_vel(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cmd: StylePhaseCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    _, ref_jv = cmd.ref_joints()
    err_sq = ((asset.data.joint_vel[:, cmd.joint_ids] - ref_jv) ** 2).sum(dim=-1)
    return _style_gate(env, min_cmd=STYLE_TROT_GATE_MIN_CMD) * torch.exp(-STYLE_TROT_SCALE_JOINT_VEL * err_sq)


def style_trot_root_h(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cmd: StylePhaseCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    err = (asset.data.root_pos_w[:, 2] - cmd.ref_h()).abs()
    return _style_gate(env, min_cmd=STYLE_TROT_GATE_MIN_CMD) * torch.exp(-STYLE_TROT_SCALE_ROOT_H * err)


def style_trot_root_rp(env, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    cmd: StylePhaseCommand = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    quat = asset.data.root_quat_w
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    roll = torch.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2 * (w * y - z * x), -1.0, 1.0))
    ref_roll, ref_pitch = cmd.ref_rp()
    err = (roll - ref_roll).abs() + (pitch - ref_pitch).abs()
    return _style_gate(env, min_cmd=STYLE_TROT_GATE_MIN_CMD) * torch.exp(-STYLE_TROT_SCALE_ROOT_RP * err)


@configclass
class UnitreeB2StyleTaskTrotEnvCfg(UnitreeB2StyleTaskEnvCfg):
    """v14-гибрид (медленный шаг) + ВТОРОЙ style-термин от rescale15-клипа
    рыси, активный только на |cmd|>=0.7. Родитель уже даёт: walk_reset
    task-награду, slow-walk style (гейт >=0.1, thigh x2), windowed adaptive
    phase-matching с монотонностью, 8192 сред."""

    def __post_init__(self):
        super().__post_init__()

        # ------Commands: второй трекер, БЕЗ телепорта (владеет slow-walk)------
        self.commands.style_trot_phase = StylePhaseCommandCfg(
            motion_json_path=STYLE_TROT_MOTION_JSON,
            window_tempo_scale=STYLE_TROT_WINDOW_TEMPO_SCALE,
            enable_rsi_teleport=False,
            joint_pos_weight_multipliers={},  # UNIFORM старт (design: измерить прежде чем весить)
        )
        # НЕ добавляем в observations -- обучение видит те же 47 obs, что
        # v14 (обратная совместимость bench/onnx, sidecar не меняется).
        # Trot-трекер работает "невидимо" для политики, только как
        # источник reward-сигнала (аналогично тому, что _style_gate уже
        # невидим для политики, только модулирует reward).

        # ------Rewards: style_trot термы поверх уже существующих slow-walk------
        self.rewards.style_trot_joint_pos = RewTerm(
            func=style_trot_joint_pos, weight=STYLE_TROT_WEIGHT_JOINT_POS,
            params={"command_name": "style_trot_phase", "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.style_trot_joint_vel = RewTerm(
            func=style_trot_joint_vel, weight=STYLE_TROT_WEIGHT_JOINT_VEL,
            params={"command_name": "style_trot_phase", "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.style_trot_root_h = RewTerm(
            func=style_trot_root_h, weight=STYLE_TROT_WEIGHT_ROOT_H,
            params={"command_name": "style_trot_phase", "asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.style_trot_root_rp = RewTerm(
            func=style_trot_root_rp, weight=STYLE_TROT_WEIGHT_ROOT_RP,
            params={"command_name": "style_trot_phase", "asset_cfg": SceneEntityCfg("robot")},
        )
        # НЕ вызываем disable_zero_weight_rewards() снова -- родитель
        # (UnitreeB2StyleTaskEnvCfg.__post_init__) уже вызвал его в конце
        # super().__post_init__(), повторный вызов падает на уже
        # обнулённых (None) термах предыдущего прохода (AttributeError:
        # 'NoneType' object has no attribute 'weight' -- поймано на первом
        # запуске). Новые style_trot_* термы все с ненулевым весом, им
        # это и не требовалось.
