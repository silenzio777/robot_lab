# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""WALK_RESET -- систематический возврат к проверенному минимальному рецепту,
не продолжение накопленной сложности `walk_env_cfg.py`.

Заказ хозяина 2026-08-30 после полутора недель без системного успеха на
JUMP v10/REAR_STAND/handstand: "давай спокойно всё проанализируем и сделаем
новый сетап с правильным подходом". Полный разбор — `~/base/DISTILLED/
2026-08-30_walk-reset-vs-rudin-baseline.md`. Источник рецепта: Rudin,
Hoeller, Reist, Hutter, "Learning to Walk in Minutes Using Massively
Parallel Deep Reinforcement Learning" (CoRL 2021, arXiv:2109.11978) --
ТОТ САМЫЙ корень, из которого растёт весь наш `robot_lab`
(legged_gym -> IsaacGym -> этот форк на IsaacLab). Числа их Table 2/3
прочитаны напрямую из PDF (`WHITEPAPERS/2109.11978v3...`), не по памяти.

КЛЮЧЕВОЙ ПРОВЕРЕННЫЙ ФАКТ (не оценка): `IsaacLab`'s `RewardManager.compute()`
(`isaaclab/managers/reward_manager.py:141`) считает `value = term(env) *
weight * dt` -- ТА ЖЕ формула, что в таблице статьи (вес записан как "1dt").
Значит голые числа веса у них и у нас -- одна единица измерения, сравнение
ниже точное.

## Что показало точное сравнение (`rough_env_cfg.py` -- база под WALK -- vs Table 2 статьи)

- `feet_air_time` -- у НАС выключен на уровне `rough_env_cfg.py` (weight=0),
  у НИХ это ВТОРОЙ по величине терм во всей таблице (2.0, уступает только
  tracking'у скорости). Не мелкий регулятор -- почти равный по важности
  самому tracking'у. Наша собственная WALK-сага ("30x слишком слабый feet_
  air_time") теперь имеет точное числовое объяснение.
- `joint_acc_l2` -- у нас в ~10000 раз слабее их суммарного joint_motion
  терма (vel+acc вместе).
- Никакого `upward`-эквивалента (штраф за отклонение от плоской ориентации)
  в их рецепте НЕТ ВООБЩЕ.
- Никакой gait-phase/synchrony машинерии в их рецепте НЕТ ВООБЩЕ -- цитата
  статьи (раздел 3.2): "neither the reward function nor the action space
  has any gait-dependent elements". Их policy ВСЕГДА сходится к настоящему
  троту сама, без единой строчки, которая бы это навязывала.

## Дизайн-решение этого файла

Не трогаю `walk_env_cfg.py` (там осталась вся история находок, полезная
если этот заход не сработает) -- НОВЫЙ, минимальный файл поверх
`UnitreeB2RoughEnvCfg` напрямую. Одна главная гипотеза за раз (закон
стокового рецепта, уже установленный в проекте): честно взвешенный
`feet_air_time` + стоковый `feet_gait` (Go2-валидированное число,
0.5) МОГУТ оказаться достаточными сами по себе, без всей
`GaitPhaseCommand`/`walk_periodic_contact_suggestion`/`walk_pair_match`
машинерии, которую строили несколько ночей. Если гипотеза не
подтвердится -- у нас уже есть готовый, куда более сложный `walk_env_cfg.py`
как fallback, ничего не потеряно.

НЕ включено сюда (сознательно, не забыто): изменение PD-гейнов. Rudin для
A1 (ЛЕГЧЕ ANYmal) их СНИЗИЛ -- у нас B2 ТЯЖЕЛЕЕ ANYmal, направление
изменения неочевидно, копировать "в лоб" рискованно без отдельного
разбора. Оставлено train/хозяину как отдельный вопрос."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeB2RoughEnvCfg

# --- Пропорции взяты из Table 2 статьи, применены к НАШИМ уже существующим
# --- track_lin_vel_xy_exp=3.0 (не их абсолютным числам -- единицы симуляции
# --- не гарантированно идентичны, отношение безопаснее переносить, чем
# --- голое число).
#
# feet_air_time : track_lin_vel = 2.0 : 1.0 у них -> при нашем 3.0 это 6.0.
# Наше текущее (в walk_env_cfg.py) WALK-значение было 3.0 -- вдвое меньше
# пропорции рецепта.
WALK_RESET_FEET_AIR_TIME_WEIGHT = 6.0

# joint_motion (vel+acc вместе) : track_lin_vel = 0.001 : 1.0 у них -> при
# нашем 3.0 это 0.003 суммарно. Делим пополам между vel/acc как первая
# прикидка (у них это один терм на двоих, у нас -- два отдельных слота) --
# ПОМЕЧЕНО как не откалибровано, первая оценка, не измеренное число.
#
# -0.0015 -> differentiated fix (2026-08-30, ~30мин после первого запуска,
# train+base совместная диагностика): первый прогон (it0-27) показал НЕ
# холодный старт, стабилизирующийся сам -- joint_acc_l2 РОС по модулю
# (it0=-1212 -> it24=-8420), Mean value_function loss доехал до ~15
# МИЛЛИОНОВ, Mean reward -146k..-218k и ухудшался по итерациям, при этом
# track_lin_vel_xy_exp оставался крошечным (0.10-0.17). Episode length
# РОС (539->659) -- термин не убивал эпизоды через termination, он ломал
# value function изнутри, не мгновенный крах.
#
# base's точный пересчёт (episode-накопленная сумма УЖЕ weight*dt --
# rsl_rl/IsaacLab's episode_sums -- значит dt сокращается в отношении,
# веса сравнимы напрямую): joint_acc_l2/track_lin_vel = -8420/0.13 ~=
# 64800x -- совпадает с изначальной прикидкой 50000-70000x. Обратный
# расчёт сырого (до веса) Σ‖q̈‖² ~= 8420/0.0015 ~= 5.6 МИЛЛИОНА за эпизод
# -- полный паритет с tracking потребовал бы веса ~-2.3e-8. Первый,
# более осторожный шаг: -1.5e-7 (10000x уменьшение, не 10x -- та
# самая staged-эскалация discipline, что весь день использовалась для
# jump_v10_launch_thigh_hold, НО base's явное предупреждение: ситуация
# на 4-5 порядков хуже launch_thigh_hold-случая (перекос там был
# ~2:1-1.17:1, здесь ~65000:1) -- умеренный ШАГ (не обнуление) при
# таком перекосе всё ещё означает БОЛЬШОЙ множитель, не 10x). Оставляет
# joint_acc_l2 ~6.5x над tracking -- реальное давление на плавность,
# но не смотрит сигнал. Если после 20-30 итераций value_function
# loss/reward не стабилизируется -- второй шаг к полному паритету
# -2.3e-8, не гадание, а прямое следствие того же теста.
#
# joint_vel_l2 -- ПЕРВЫЙ отдельный запрос base: не переносить тот же
# множитель на velocity вслепую, ускорение и скорость структурно
# разного масштаба (acc ~ Δvel/dt, при малом dt акс-термы систематически
# больше даже для той же физической "резкости"). Измерено отдельно:
# joint_vel_l2 (тот же период, тот же исходный вес -0.0015) it0=-0.0069
# -> it27=-0.3126 -- растёт, но отношение к track_lin_vel максимум ~2x,
# НЕ похоже на acc-класс перекоса. Оставлен на исходном -0.0015, не
# трогаю -- в разумных пределах, отдельная проверка не показала той же
# патологии.
#
# Методологический урок (base, для будущих новых термов): перенос веса
# через пропорцию из ДРУГОЙ статьи/кодовой базы рискованнее эмпирической
# калибровки по факту измерения (то, что rear_stand/JUMP v10 делали
# правильно весь этот проект) -- сырая величина зависит от деталей
# конкретной симуляции (единицы, decimation, PD-гейны), пропорция весов
# между источниками не гарантирует пропорцию реальных вкладов. Для
# следующего нового терма -- сначала замерить сырое значение на одной
# итерации, потом уже назначать вес, не наоборот.
#
# Остановлено и перезапущено с нуля (ни первый, ни промежуточный 10x-
# прогон не дали спасаемого позднего чекпоинта -- save_interval=100,
# оба остановлены до первого сохранения).
WALK_RESET_JOINT_VEL_WEIGHT = -0.0015
WALK_RESET_JOINT_ACC_WEIGHT = -1.5e-7

# upward: у рецепта такого терма нет вообще (наше текущее значение 3.0 --
# собственное добавление, не из источника). Не убираю до нуля без замера
# (могло подавлять реальную проблему, которую мы не видим) -- символическое
# значение, сильно ниже текущего, чтобы посмотреть, всплывёт ли тот самый
# "перекос корпуса/топающая нога", про который сама статья пишет как про
# типичный артефакт недонастроенного веса (раздел 4.2, "there are often
# artifacts... After tuning of the reward weights...").
WALK_RESET_UPWARD_WEIGHT = 0.5


@configclass
class UnitreeB2WalkResetEnvCfg(UnitreeB2RoughEnvCfg):
    """Минимальный WALK-рецепт, дисциплинированный возврат к проверенной
    базе -- см. модульный docstring для полного обоснования каждого числа."""

    def __post_init__(self):
        super().__post_init__()

        # ------------------------------Rewards: честный feet_air_time------------------------------
        # Формула уже бит-в-бит совпадает с рецептом (проверено напрямую,
        # mdp.rewards.feet_air_time: `(last_air_time - threshold) *
        # first_contact`, тот же Σ(t_air,f - 0.5), что в статье) -- меняем
        # только вес и порог, не саму механику.
        self.rewards.feet_air_time.weight = WALK_RESET_FEET_AIR_TIME_WEIGHT
        # threshold=0.5 -- СТОКОВОЕ значение статьи/rough_env_cfg.py, НЕ
        # 0.2 из walk_env_cfg.py (то было подогнано специально под
        # phase-clock's duty_factor=0.65@1.4Hz -- здесь phase-clock нет,
        # нет причины отклоняться от проверенного 0.5).
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]

        # ------------------------------Rewards: стоковый feet_gait, БЕЗ phase-clock------------------------------
        # Go2-валидированное число (0.5), стоковые max_err/std (не
        # экспериментально расширенные walk_env_cfg.py's own probe values) --
        # гипотеза этого файла: честный feet_air_time + это само по себе
        # достаточно, без кастомной синхронизации.
        self.rewards.feet_gait.weight = 0.5
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (
            ("FL_calf", "RR_calf"), ("FR_calf", "RL_calf")
        )

        # ------------------------------Rewards: joint_acc/vel ближе к пропорции рецепта------------------------------
        self.rewards.joint_vel_l2.weight = WALK_RESET_JOINT_VEL_WEIGHT
        self.rewards.joint_vel_l2.params["asset_cfg"].joint_names = ".*"
        self.rewards.joint_acc_l2.weight = WALK_RESET_JOINT_ACC_WEIGHT

        # ------------------------------Rewards: upward снижен, не убран------------------------------
        self.rewards.upward.weight = WALK_RESET_UPWARD_WEIGHT

        # ------------------------------Rewards: feet_slide, стоковое значение------------------------------
        # Тот же аргумент, что и walk_env_cfg.py уже установил (не про
        # gait-phase, общий anti-drag регулятор) -- оставляем.
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]

        # ------------------------------Events: широкая рандомизация массы (не про gait-phase, оставляем)------------------------------
        # Тот же фикс, что уже был найден в walk_env_cfg.py 2026-08-26 --
        # per-body распределение по всем 13 телам, не только base_link.
        self.events.randomize_rigid_body_mass.params["asset_cfg"].body_names = [
            self.base_link_name
        ] + [f"{leg}_{part}" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf")]
        self.events.randomize_rigid_body_mass.params["mass_distribution_params"] = (0.7, 1.3)
        self.events.randomize_rigid_body_mass.params["operation"] = "scale"

        # ------------------------------Curriculum: скоростной, не про gait-phase, оставляем------------------------------
        self.curriculum.command_levels = CurrTerm(
            func=mdp.command_levels_vel,
            params={"reward_term_name": "track_lin_vel_xy_exp", "range_multiplier": (0.2, 1.0)},
        )

        # ------------------------------НЕ подключено сознательно------------------------------
        # GaitPhaseCommand / walk_periodic_contact_suggestion / walk_pair_match
        # -- вся кастомная gait-phase машинерия walk_env_cfg.py. Рецепт статьи
        # обходится без неё; это ГИПОТЕЗА для проверки, не утверждение, что
        # машинерия не нужна -- если этот заход не даст настоящий
        # диагональный трот (проверять gate-системой base/train, та же
        # методология что на JUMP v10), возвращаемся к walk_env_cfg.py как
        # более сложному, но уже частично проверенному fallback.

        if self.__class__.__name__ == "UnitreeB2WalkResetEnvCfg":
            self.disable_zero_weight_rewards()
