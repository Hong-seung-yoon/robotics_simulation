#!/bin/bash
# raw40 annotate가 끝나면 생성 시작 직후 중단 → 조건 A seed101 실패 기록(seed 42) → raw40 생성 재개. (sim 동시 1개 규칙)
cd ~/IsaacLab_RS; source ~/anaconda3/etc/profile.d/conda.sh; conda activate lerobot-arena; export PYTHONUNBUFFERED=1
L=logs/hw2
until grep -q "Exported [0-9]* (out of" $L/ta20own20_raw_annotate.log 2>/dev/null; do sleep 10; done
until [ -e datasets/ta20own20_raw_mimic_raw.hdf5 ] || pgrep -f "ta20own20_raw_mimic_raw.hdf5" >/dev/null; do sleep 2; done
pkill -9 -f "hw2_scripts/pipeline.sh datasets/toy2box_ta20_own20_raw"; pkill -f "output_file datasets/ta20own20_raw_mimic_raw"
while pgrep -f "output_file datasets/ta20own20_raw_mimic_raw" >/dev/null; do sleep 2; done
rm -f datasets/ta20own20_raw_mimic_raw.hdf5 datasets/ta20own20_raw_mimic_raw_failed.hdf5
echo "paused raw40 before generation $(date)"
CK=logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0/bc_rnn_low_dim_agibot_toy2box/20261019184411/models/model_epoch_2000.pth
./isaaclab.sh -p ~/IsaacLab_RS_hw2/hw2_scripts/play_record.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --checkpoint $CK \
  --headless --seed 42 --num_rollouts 100 --horizon 800 --record_dir $L/failrec_condA_seed101_eval42 > $L/failrec_condA_seed101_eval42.log 2>&1
grep "Success rate" $L/failrec_condA_seed101_eval42.log
python -I ~/IsaacLab_RS_hw2/hw2_scripts/classify_failures.py $L/failrec_condA_seed101_eval42 --csv $L/failrec_condA_seed101_eval42.csv
echo "restarting raw40 pipeline $(date)"
nohup ~/IsaacLab_RS_hw2/hw2_scripts/pipeline.sh datasets/toy2box_ta20_own20_raw.hdf5 ta20own20_raw 1000 > $L/pipeline_ta20own20_raw.out 2>&1 &
