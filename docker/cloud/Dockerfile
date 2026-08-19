# Cloud training image for B2 skills (jump / rear_stand / leg_lift / walk).
#
# Security-by-construction: everything this image is built FROM is a fresh
# clone of PUBLIC upstream repos (isaac-sim/IsaacLab, fan-ziqi/robot_lab) --
# never our private fork's working tree or git history. The ONLY private
# content baked in is docker/cloud/b2_overlay/ (regenerate it locally first
# via prepare_overlay.sh) -- 11 task-config files, no .git, no .md, no
# CLAUDE.md, no agent memory. Nothing else from our machine goes in.
#
# Pinned versions match our locally-validated stack (NOT the IsaacLab repo's
# own docker defaults, which currently point at isaac-sim 5.0.0 -- untested
# by us):
#   IsaacLab      2.2.0
#   Isaac Sim     4.5.0
#   torch         2.7.0+cu128 / torchvision 0.22.0+cu128
#   rsl-rl-lib    2.3.3
#   gymnasium     1.2.0
#   Ubuntu        22.04 (inherited from the isaac-sim base image)
#
# Build context is docker/cloud/ ONLY (not the repo root) -- this is
# deliberate: the build context is what actually gets sent to the docker
# daemon (locally, or transferred to the remote host to build there), and
# it must never be the whole private repo. Run prepare_overlay.sh first,
# then build from inside docker/cloud/:
#   cd docker/cloud && docker build -t b2-cloud-train .
#
# Run (single GPU, results land in ./cloud_logs on the host):
#   docker run --rm -it --gpus all \
#     -v $(pwd)/cloud_logs:/workspace/robot_lab/logs \
#     b2-cloud-train \
#     ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
#       --task Isaac-JumpB2-v0 --headless

ARG ISAACSIM_VERSION=4.5.0
FROM nvcr.io/nvidia/isaac-sim:${ISAACSIM_VERSION} AS isaac-base

ENV ACCEPT_EULA=Y \
    DEBIAN_FRONTEND=noninteractive \
    ISAACLAB_PATH=/workspace/isaaclab \
    ROBOT_LAB_PATH=/workspace/robot_lab

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

# --- IsaacLab: fresh clone of the PUBLIC upstream repo, pinned tag ---
RUN git clone --depth 1 --branch v2.2.0 \
        https://github.com/isaac-sim/IsaacLab.git ${ISAACLAB_PATH}
RUN ln -sf /isaac-sim ${ISAACLAB_PATH}/_isaac_sim

WORKDIR ${ISAACLAB_PATH}
RUN pip install --upgrade pip==23 setuptools==65 \
    && pip install "cmake<3.27"
ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
RUN pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
RUN ./isaaclab.sh --install

# --- robot_lab: fresh clone of the PUBLIC upstream repo (fan-ziqi/robot_lab) ---
RUN git clone --depth 1 https://github.com/fan-ziqi/robot_lab.git ${ROBOT_LAB_PATH}

# --- Overlay: ONLY our 11 B2 task-config files, no history, no docs ---
# Regenerate locally with docker/cloud/prepare_overlay.sh before building.
COPY b2_overlay/ ${ROBOT_LAB_PATH}/

WORKDIR ${ROBOT_LAB_PATH}
RUN pip install -e .

WORKDIR ${ISAACLAB_PATH}
ENTRYPOINT ["/bin/bash"]
