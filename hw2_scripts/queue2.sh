#!/bin/bash
# B(진행 중) 종료 대기 → noMimic 100 평가(학습은 별도 선행) → F → E1 ; sim 동시 1개
cd ~/IsaacLab_RS; source ~/anaconda3/etc/profile.d/conda.sh; conda activate lerobot-arena; export PYTHONUNBUFFERED=1
L=logs/hw2; H=~/IsaacLab_RS_hw2/hw2_scripts; BP=$1
while kill -0 $BP 2>/dev/null; do sleep 30; done; echo "B done $(date)"; tail -2 $L/pipeline_own18_raw_m192.out
echo "noMimic eval start $(date)"; while pgrep -f "train_seed.sh 1[01]1 ./datasets/own20_sel80_raw_annotated" >/dev/null; do sleep 30; done
for S in 101 111; do
  CK=$(ls -d logs/robomimic/*/own100_nomimic_seed$S/*/models | tail -1)/model_epoch_2000.pth
  ls $CK || { echo "ABORT missing $CK"; continue; }
  ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --checkpoint $CK \
    --headless --seed 42 --num_rollouts 100 --horizon 800 > $L/own100_nomimic_seed${S}_eval42.log 2>&1
  echo "own100_nomimic seed$S: $(grep 'Success rate' $L/own100_nomimic_seed${S}_eval42.log)"
done
echo "F start $(date)"; $H/pipeline.sh datasets/own45_raw_m1000_annotated.hdf5 own45_raw_m1000 1000 > $L/pipeline_own45_raw_m1000.out 2>&1; tail -2 $L/pipeline_own45_raw_m1000.out
echo "E1 start $(date)"; $H/pipeline.sh datasets/ta20_raw_m1000_annotated.hdf5 ta20_raw_m1000 1000 > $L/pipeline_ta20_raw_m1000.out 2>&1; tail -2 $L/pipeline_ta20_raw_m1000.out
