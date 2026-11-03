#!/bin/bash
# usage: train_cfg.sh CONFIG.json DATASET NAME  (registered json은 건드리지 않음)
cd ~/IsaacLab_RS; source ~/anaconda3/etc/profile.d/conda.sh; conda activate lerobot-arena; export PYTHONUNBUFFERED=1
./isaaclab.sh -p ~/IsaacLab_RS_hw2/hw2_scripts/train_cfg.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --algo bc \
  --config "$1" --dataset "$2" --name "$3" > logs/robomimic/train_$3.log 2>&1
