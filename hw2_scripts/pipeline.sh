#!/bin/bash
# usage: pipeline.sh SRC.hdf5 TAG N_SUCCESS   (예: datasets/toy2box_ta20_own20_trim10.hdf5 ta20own20_trim10 1000)
# 주의: RmpFlow가 reset마다 설정 yaml을 공용 경로에 재다운로드하므로 sim 프로세스는 동시에 하나만 돌릴 것
# annotate --auto → Mimic 생성(성공 N개) → N개로 자르기 → seed 101/111 학습 → seed 42 평가
set -e
SRC=$1; TAG=$2; N=$3
cd ~/IsaacLab_RS; source ~/anaconda3/etc/profile.d/conda.sh; conda activate lerobot-arena
export PYTHONUNBUFFERED=1
H=~/IsaacLab_RS_hw2/hw2_scripts; L=logs/hw2; mkdir -p $L
ANN=datasets/${TAG}_annotated.hdf5; GEN=datasets/${TAG}_mimic_raw.hdf5; OUT=datasets/${TAG}_mimic${N}.hdf5
[ -f $ANN ] || ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 \
  --auto --input_file $SRC --output_file $ANN --headless > $L/${TAG}_annotate.log 2>&1
python -I ~/IsaacLab_RS_hw2/summarize_demos.py $ANN | head -2
[ -f $GEN ] || ./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 \
  --num_envs 100 --generation_num_trials $N --input_file $ANN --output_file $GEN --headless > $L/${TAG}_generate.log 2>&1
$H/finish.sh $TAG $N
