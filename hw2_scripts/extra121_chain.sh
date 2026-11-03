#!/bin/bash
# 추가(필수 아님): seed 121 학습 ∥ seed101/111 실패 분류(sim) → seed 121 평가(sim, 순차)
cd ~/IsaacLab_RS; source ~/anaconda3/etc/profile.d/conda.sh; conda activate lerobot-arena; export PYTHONUNBUFFERED=1
H=~/IsaacLab_RS_hw2/hw2_scripts; L=logs/hw2; B=logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
$H/train_seed.sh 121 ./datasets/own20_sel80_raw_mimic1000.hdf5 own20_sel80_raw_seed121 & TP=$!
$H/failrec.sh $B/own20_sel80_raw_seed101/20261029220420/models/model_epoch_2000.pth $B/own20_sel80_raw_seed111/20261029220620/models/model_epoch_2000.pth
wait $TP
CK=$(ls -d $B/own20_sel80_raw_seed121/*/models | tail -1)/model_epoch_2000.pth
ls $CK || { echo "ABORT: missing $CK"; exit 1; }
grep '"seed"' $(dirname $(dirname $CK))/config.json
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
  --checkpoint $CK --headless --seed 42 --num_rollouts 100 --horizon 800 > $L/own20_sel80_raw_seed121_eval42.log 2>&1
echo "own20_sel80_raw seed121: $(grep 'Success rate' $L/own20_sel80_raw_seed121_eval42.log)"
