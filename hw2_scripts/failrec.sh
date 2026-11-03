#!/bin/bash
# usage: failrec.sh TAG CKPT... (순차 실행; sim 동시 1개)
cd ~/IsaacLab_RS; source ~/anaconda3/etc/profile.d/conda.sh; conda activate lerobot-arena; export PYTHONUNBUFFERED=1; L=logs/hw2
for CK in "$@"; do
  T=$(echo $CK | sed -E 's|.*/([^/]+)/[0-9]+/models/.*|\1|')
  ./isaaclab.sh -p ~/IsaacLab_RS_hw2/hw2_scripts/play_record.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --checkpoint $CK \
    --headless --seed 42 --num_rollouts 100 --horizon 800 --record_dir $L/failrec_${T}_eval42 > $L/failrec_${T}_eval42.log 2>&1
  echo "== $T $(grep 'Success rate' $L/failrec_${T}_eval42.log)"
  python -I ~/IsaacLab_RS_hw2/hw2_scripts/classify_failures.py $L/failrec_${T}_eval42 --approach 0.051 --csv $L/failrec_${T}_eval42.csv
done
