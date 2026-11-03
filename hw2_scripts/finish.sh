#!/bin/bash
# usage: finish.sh TAG N [WAIT_PID]  — 생성 완료 후: N개로 자르기 → seed 101/111 학습(병렬) → seed 42 평가(순차; sim 동시 실행 금지)
TAG=$1; N=$2; WPID=$3
cd ~/IsaacLab_RS; source ~/anaconda3/etc/profile.d/conda.sh; conda activate lerobot-arena
export PYTHONUNBUFFERED=1
H=~/IsaacLab_RS_hw2/hw2_scripts; L=logs/hw2
[ -n "$WPID" ] && while kill -0 $WPID 2>/dev/null; do sleep 30; done
GEN=datasets/${TAG}_mimic_raw.hdf5; OUT=datasets/${TAG}_mimic${N}.hdf5
grep -o "[0-9]*/[0-9]* ([0-9.]*%) successful demos" $L/${TAG}_generate.log | tail -1
python -I $H/cap_demos.py $GEN $OUT $N | tee /dev/stderr | grep -q " -> $N demos" || { echo "ABORT: fewer than $N successes"; exit 1; }
for S in 101 111; do $H/train_seed.sh $S ./$OUT ${TAG}_seed$S & sleep 120; done; wait
for S in 101 111; do
  CK=$(ls -d logs/robomimic/*/${TAG}_seed$S/*/models | tail -1)/model_epoch_2000.pth
  ls $CK || { echo "ABORT: missing $CK"; exit 1; }
  ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
    --checkpoint $CK --headless --seed 42 --num_rollouts 100 --horizon 800 > $L/${TAG}_seed${S}_eval42.log 2>&1
  echo "$TAG seed$S: $(grep 'Success rate' $L/${TAG}_seed${S}_eval42.log)"
done
