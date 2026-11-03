#!/bin/bash
# sim 순서: F+rnn50 s101→s111 (rule-safe) → 학습 끝난 변형 s101 평가 → s111 평가
cd ~/IsaacLab_RS; L=logs/hw2; H=~/IsaacLab_RS_hw2/hw2_scripts; B=logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
ev(){ while pgrep -f "train_cfg.py.*--name $1( |$)" >/dev/null; do sleep 60; done; echo "eval $1 $(date)"; $H/failrec.sh $(ls -d $B/$1/*/models | tail -1)/model_epoch_2000.pth; }
for T in own45m1000_rnn50_seed101 own45m1000_rnn50_seed111 own100m1000_rnn50norm_seed101 own100m1000_relrnn50_seed101 own100m1000_rnn50lrdecay_seed101 own100m1000_rnn50norm_seed111 own100m1000_relrnn50_seed111 own100m1000_rnn50lrdecay_seed111; do ev $T; done
echo "queue7 done $(date)"
