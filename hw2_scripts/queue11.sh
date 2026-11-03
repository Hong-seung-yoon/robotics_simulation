#!/bin/bash
# noisyeval s101 종료 대기 → Mimic 추가 생성(seed 2, 성공 1000) → 트럭 x 균형 1000개 → ②설정 학습 → 평가들
cd ~/IsaacLab_RS; source ~/anaconda3/etc/profile.d/conda.sh; conda activate lerobot-arena; export PYTHONUNBUFFERED=1
L=logs/hw2; H=~/IsaacLab_RS_hw2/hw2_scripts; B=logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
ev(){ while pgrep -f "train_cfg.py.*--name $1( |$)" >/dev/null; do sleep 60; done; echo "eval $1 $(date)"; $H/failrec.sh $(ls -d $B/$1/*/models | tail -1)/model_epoch_2000.pth; }
while kill -0 $1 2>/dev/null; do sleep 30; done; echo "noisyeval done $(date)"
echo "gen pool2 start $(date)"
HW2_GEN_SEED=2 ./isaaclab.sh -p $H/generate_dataset_seed.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 --num_envs 100 --generation_num_trials 1000 --input_file datasets/own20_sel80_raw_annotated.hdf5 --output_file datasets/own20_sel80_raw_mimic_pool2.hdf5 --headless > $L/own20_sel80_pool2_generate.log 2>&1
echo "gen pool2 done $(date): $(grep -ao '[0-9]*/[0-9]* ([0-9.]*%) successful demos' $L/own20_sel80_pool2_generate.log | tail -1)"
python -I $H/balance_truck_x.py datasets/own20_sel80_balx_mimic1000.hdf5 datasets/own20_sel80_balx_mimic1000_map.csv 1000 datasets/own20_sel80_raw_mimic_raw.hdf5 datasets/own20_sel80_raw_mimic_pool2.hdf5
for S in 101 111; do python3 -c "
import json;c=json.load(open('$H/cfg_rnn50_seed101.json'));c['train']['seed']=$S;json.dump(c,open('$H/cfg_balx_rnn50_seed$S.json','w'),indent=4)"
 nohup $H/train_cfg.sh $H/cfg_balx_rnn50_seed$S.json ./datasets/own20_sel80_balx_mimic1000.hdf5 own100m1000_balx_rnn50_seed$S >/dev/null 2>&1 &
done; sleep 60
for T in rnn50norm rnn50lrdecay; do ev own100m1000_${T}_seed101; done
ev own100m1000_balx_rnn50_seed101; ev own100m1000_balx_rnn50_seed111
for T in relrnn50 rnn100 rnn50len800 rnn50noise; do ev own100m1000_${T}_seed101; done
echo "queue11 done $(date)"
