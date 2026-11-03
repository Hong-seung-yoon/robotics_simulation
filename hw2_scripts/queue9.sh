#!/bin/bash
# queue8 종료 대기 → obs noise 변형 평가
cd ~/IsaacLab_RS; H=~/IsaacLab_RS_hw2/hw2_scripts; B=logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
ev(){ while pgrep -f "train_cfg.py.*--name $1( |$)" >/dev/null; do sleep 60; done; echo "eval $1 $(date)"; $H/failrec.sh $(ls -d $B/$1/*/models | tail -1)/model_epoch_2000.pth; }
while pgrep -f "hw2_scripts/queue8" >/dev/null; do sleep 60; done
ev own100m1000_rnn50noise_seed101; ev own100m1000_rnn50noise_seed111
echo "queue9 done $(date)"
