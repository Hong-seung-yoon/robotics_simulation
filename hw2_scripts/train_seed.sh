#!/bin/bash
# usage: train_seed.sh SEED DATASET NAME   (run from ~/IsaacLab_RS)
# 조건 A와 동일(bc_rnn_low_dim.json, 2000 epoch), seed만 변경. 원본 json은 .orig로 백업.
set -e
SEED=$1; DATA=$2; NAME=$3
cd ~/IsaacLab_RS; source ~/anaconda3/etc/profile.d/conda.sh; conda activate lerobot-arena
J=source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/place/config/agibot/agents/robomimic/bc_rnn_low_dim.json
[ -f $J.orig ] || cp $J $J.orig
exec 9>/tmp/hw2_json.lock; flock 9
python3 - "$J" "$SEED" <<'PY'
import json,sys
p,s=sys.argv[1],int(sys.argv[2]); c=json.load(open(p+'.orig')); c['train']['seed']=s
json.dump(c,open(p,'w'),indent=4)
PY
mkdir -p logs/robomimic
LOG=logs/robomimic/train_${NAME}.log
PYTHONUNBUFFERED=1 ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --algo bc --dataset "$DATA" --name "$NAME" > $LOG 2>&1 &
PID=$!
# 학습이 config를 읽을 때까지 대기 후 원본 복원
for i in $(seq 600); do ls logs/robomimic/*/$NAME/*/config.json >/dev/null 2>&1 && break; sleep 1; done
cp $J.orig $J; flock -u 9
wait $PID
