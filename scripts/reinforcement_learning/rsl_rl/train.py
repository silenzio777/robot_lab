# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import cli_args

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import torch
from datetime import datetime

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # save resume path before creating a new log_dir
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        # --no_resume_optimizer (see cli_args.py's own comment): cross-task/
        # cross-economy warm-starts (e.g. LegLift Stage 0 -> Stage 1) want the
        # donor's weights but NOT its Adam optimizer moments, which are tuned
        # for a different reward landscape.
        runner.load(resume_path, load_optimizer=not args_cli.no_resume_optimizer)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # run training
    # --- ПРОТОКОЛ ЗАЩИТЫ ПЕРЕСАДКИ (train+base 2026-09-03, только при
    # --- STYLE_FREEZE_ACTOR=1 в окружении; обычные раны не затронуты) ---
    # Свежий критик + пересаженный актор: шумные advantages разрушают
    # актора до сходимости критика (v6: телепорт-tilt 10.3->87.2 за 400
    # ит). Заморозка через requires_grad=False (Л1 base: lr=0 затирается
    # adaptive-KL планировщиком); на разморозке принудительный сброс lr к
    # базовому (Л2: за время заморозки KL~0 разгоняет lr до потолка 1e-2).
    # Критерий разморозки: explained_variance > 0.7 стабильно 50 ит, кап
    # 1000 (протокол обновлён base 2026-09-03: абсолютный value_loss<0.05
    # был откалиброван под нормальную шкалу reward и недостижим при |R|~200;
    # ev безразмерен: 1 - Var(return-value)/Var(return) на батче storage,
    # считается ДО update -- после него storage очищается).
    import os as _os
    if _os.environ.get("STYLE_FREEZE_ACTOR") == "1":
        _ac = runner.alg.policy if hasattr(runner.alg, "policy") else runner.alg.actor_critic
        _actor_params = list(_ac.actor.parameters())
        if hasattr(_ac, "std") and isinstance(_ac.std, torch.nn.Parameter):
            _actor_params.append(_ac.std)
        for _p in _actor_params:
            _p.requires_grad_(False)
        print(f"[FREEZE] актор заморожен ({sum(p.numel() for p in _actor_params)} параметров), критик-вармап")
        _orig_update = runner.alg.update
        _fs = {"frozen": True, "stable": 0, "it": 0}

        def _update_with_freeze():
            _ev = 0.0
            _st = getattr(runner.alg, "storage", None)
            if _st is not None and getattr(_st, "returns", None) is not None:
                _ret = _st.returns.flatten()
                _var = torch.var(_ret)
                if _var > 1e-8:
                    _ev = float(1.0 - torch.var(_ret - _st.values.flatten()) / _var)
            loss = _orig_update()
            _fs["it"] += 1
            vl = loss.get("value_function", loss.get("value_loss", 1.0)) if isinstance(loss, dict) else loss[0]
            if _fs["frozen"]:
                _fs["stable"] = _fs["stable"] + 1 if _ev > 0.7 else 0
                if _fs["it"] % 100 == 0:
                    print(f"[FREEZE] ит {_fs['it']}: explained_variance {_ev:.3f}, stable {_fs['stable']}")
                if _fs["stable"] >= 50 or _fs["it"] >= 1000:
                    for _p in _actor_params:
                        _p.requires_grad_(True)
                    runner.alg.learning_rate = 1.0e-3
                    for _g in runner.alg.optimizer.param_groups:
                        _g["lr"] = 1.0e-3
                    _fs["frozen"] = False
                    print(f"[FREEZE] РАЗМОРОЗКА на ит {_fs['it']} (explained_variance {_ev:.3f}, value_loss {vl:.4f}, stable {_fs['stable']}), lr сброшен к 1e-3")
            return loss

        runner.alg.update = _update_with_freeze

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
