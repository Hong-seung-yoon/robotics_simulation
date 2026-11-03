#!/bin/bash
# V1 s101 분류(진행 중) 종료 대기 → ② s111 평가 → 정지 (이후 순서는 ② s111 결과와 시범 소유 답변에 따라 결정)
cd ~/IsaacLab_RS; L=logs/hw2; H=~/IsaacLab_RS_hw2/hw2_scripts; B=logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
while kill -0 $1 2>/dev/null; do sleep 30; done; echo "V1 done $(date)"
while pgrep -f "train_cfg.py.*--name own100m1000_rnn50_seed111" >/dev/null; do sleep 60; done
echo "rnn50 s111 eval $(date)"; $H/failrec.sh $(ls -d $B/own100m1000_rnn50_seed111/*/models | tail -1)/model_epoch_2000.pth
echo "queue6 done $(date)"
