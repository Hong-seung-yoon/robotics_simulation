#!/bin/bash
# F(진행 중) 종료 대기 → F seed101 실패 분류 → E1(조교20 + Mimic1000)
cd ~/IsaacLab_RS; L=logs/hw2; H=~/IsaacLab_RS_hw2/hw2_scripts; B=logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
while kill -0 $1 2>/dev/null; do sleep 30; done; echo "F done $(date)"; tail -4 $L/pipeline_own45_raw_m1000.out
C=$(ls -d $B/own45_raw_m1000_seed101/*/models | tail -1)/model_epoch_2000.pth; [ -e $C ] && $H/failrec.sh $C
echo "E1 start $(date)"; $H/pipeline.sh datasets/ta20_raw_m1000_annotated.hdf5 ta20_raw_m1000 1000 > $L/pipeline_ta20_raw_m1000.out 2>&1; tail -2 $L/pipeline_ta20_raw_m1000.out
