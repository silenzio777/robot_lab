# B2 cloud training image — build & run guide

This directory builds a self-contained Docker image for running Isaac Lab
reinforcement-learning training for a quadruped robot (Unitree B2) task set,
on a fresh cloud GPU machine.

## What's in this directory

- `Dockerfile` — builds the image.
- `b2_overlay/` — a small set of task-config source files copied into the
  image at build time (robot/task definitions, training hyperparameters).
- `sync_results.sh` — helper to pull finished checkpoints back to wherever
  you run it from, over `rsync`/`ssh`.

Nothing else in this directory is needed to build or run the image.

## Requirements on the target machine

- Ubuntu 22.04, x86_64.
- An NVIDIA GPU (tested target class: H100/H200, RTX 5090, RTX 6000) with a
  recent driver (>= 550) and `nvidia-container-toolkit` installed, so
  `docker run --gpus all` works.
- Docker Engine (20.10+) with the NVIDIA runtime configured.
- ~40 GB free disk for the built image plus training logs.
- Internet access during the build (the Dockerfile pulls the base image and
  clones two public GitHub repositories).

Quick check that GPU passthrough is working before building anything:
```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

## Build

From inside this directory (this exact directory must be the Docker build
context — do not build from a parent directory):

```bash
cd docker/cloud
docker build -t b2-cloud-train .
```

This will take a while the first time (base image pull + full Isaac Lab
install). Nothing outside this directory is sent to Docker or read during
the build.

## Run training

```bash
mkdir -p ./cloud_logs
docker run --rm -it --gpus all \
  -v "$(pwd)/cloud_logs:/workspace/robot_lab/logs" \
  b2-cloud-train \
  ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
    --task <TASK_NAME> --headless
```

Replace `<TASK_NAME>` with the Gym task id you were given (e.g. something
registered under `RobotLab-Isaac-Velocity-Rough-Unitree-B2-*`). To see the
full list of registered tasks available in the image:

```bash
docker run --rm b2-cloud-train \
  ./isaaclab.sh -p scripts/environments/list_envs.py
```

Useful `train.py` flags:
- `--headless` — no GUI, required on a server without a display.
- `--num_envs <N>` — override the number of parallel simulation environments.
- `--max_iterations <N>` — override training length.
- `--resume --load_run <run_dir> --checkpoint <file>` — resume from a
  checkpoint already present under `logs/` (e.g. one you copied in from a
  previous session).

Training writes checkpoints (`model_*.pt`), a config snapshot, and
TensorBoard event files under `logs/rsl_rl/<task>/<run_timestamp>/` inside
the container, which — because of the `-v` mount above — appears directly
on the host at `./cloud_logs/rsl_rl/<task>/<run_timestamp>/`.

To watch progress remotely:
```bash
tensorboard --logdir ./cloud_logs --bind_all
```

## Getting results back

From the machine where you want the results (not necessarily the same one
that ran the build), with SSH access to the training machine:

```bash
./sync_results.sh user@training-host [remote_path_to_robot_lab]
```

This does a read-only `rsync` pull of `model_*.pt`, `*.onnx`, `*.yaml`, and
TensorBoard event files from `<remote_path_to_robot_lab>/logs/` into
`../../cloud_results/` (relative to this directory). It never uploads
anything to the remote host.

If you'd rather do it manually:
```bash
rsync -avz user@training-host:~/robot_lab/logs/ ./cloud_results/
```

## Exporting a trained policy to ONNX (optional)

Isaac Lab / rsl_rl ship a standalone checkpoint→ONNX exporter that does not
need a GPU or a running simulation — useful for exporting on a lighter
machine after pulling the `.pt` file back:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py \
  --task <TASK_NAME> --checkpoint <path_to_model.pt> --headless
```

(consult `--help` on that script for export-only flags in your image's
version if you don't need to run a play/visualization episode).

## Notes

- The image is built entirely from public sources at build time (the base
  NVIDIA Isaac Sim image, plus two public GitHub repositories cloned fresh
  inside the Dockerfile) plus the small overlay of task-config files in
  `b2_overlay/`. There is no other project data baked into the image.
- If a build step fails on version resolution, check that the base image
  tag in `Dockerfile` and the pinned Python package versions are still
  available — NVIDIA's container registry and PyPI occasionally deprecate
  old tags/wheels.
