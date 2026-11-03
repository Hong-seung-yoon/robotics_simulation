#!/bin/bash
# F+rnn50 s101(진행 중) 종료 대기 → 변형 s101 평가 → 변형 s111 평가 → F+rnn50 s111
cd ~/IsaacLab_RS; L=logs/hw2; H=~/IsaacLab_RS_hw2/hw2_scripts; B=logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
ev(){ while pgrep -f "train_cfg.py.*--name $1( |$)" >/dev/null; do sleep 60; done; echo "eval $1 $(date)"; $H/failrec.sh $(ls -d $B/$1/*/models | tail -1)/model_epoch_2000.pth; }
while kill -0 $1 2>/dev/null; do sleep 30; done; echo "F+rnn50 s101 done $(date)"
for S in 101 111; do for V in rnn50noisyeval_PLACEHOLDER rnn50norm relrnn50 rnn50lrdecay rnn100 rnn50len800; do
  [ $V = rnn50noisyeval_PLACEHOLDER ] && V=rnn50_noisyeval; ev own100m1000_${V}_seed$S; done; done
ev own45m1000_rnn50_seed111
echo "queue8 done $(date)"
