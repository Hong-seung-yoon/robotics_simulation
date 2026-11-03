#!/bin/bash
# noisyeval → 추가 생성(seed 2) → 균등 x 데이터/학습 → norm → lrdecay → 균등 s101,s111 → postfilt s101 → 나머지 변형 s101 → ② 넘은 변형 s111
cd ~/IsaacLab_RS; source ~/anaconda3/etc/profile.d/conda.sh; conda activate lerobot-arena; export PYTHONUNBUFFERED=1
L=logs/hw2; H=~/IsaacLab_RS_hw2/hw2_scripts; B=logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
ev(){ while pgrep -f "train_cfg.py.*--name $1( |$)" >/dev/null; do sleep 60; done; echo "eval $1 $(date)"; $H/failrec.sh $(ls -d $B/$1/*/models | tail -1)/model_epoch_2000.pth; python -I $H/xbin_success.py $L/failrec_$1_eval42; }
rate(){ grep -a 'Success rate' $L/failrec_$1_eval42.log 2>/dev/null | tail -1 | awk '{print $3}'; }
while kill -0 $1 2>/dev/null; do sleep 30; done; echo "noisyeval done $(date)"
echo "gen pool2 start (HW2_GEN_SEED=2) $(date)"
HW2_GEN_SEED=2 ./isaaclab.sh -p $H/generate_dataset_seed.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 --num_envs 100 --generation_num_trials 1000 --input_file datasets/own20_sel80_raw_annotated.hdf5 --output_file datasets/own20_sel80_raw_mimic_pool2.hdf5 --headless > $L/own20_sel80_pool2_generate.log 2>&1
echo "gen pool2 done $(date): $(grep -ao '[0-9]*/[0-9]* ([0-9.]*%) successful demos' $L/own20_sel80_pool2_generate.log | tail -1)"
python -I $H/balance_truck_x.py datasets/own20_sel80_balx_mimic1000.hdf5 datasets/own20_sel80_balx_mimic1000_map.csv 1000 datasets/own20_sel80_raw_mimic_raw.hdf5 datasets/own20_sel80_raw_mimic_pool2.hdf5 || { echo ABORT balance; }
for S in 101 111; do python -c "
import json;c=json.load(open('$H/cfg_rnn50_seed101.json'));c['train']['seed']=$S;json.dump(c,open('$H/cfg_balx_rnn50_seed$S.json','w'),indent=4)"
 nohup $H/train_cfg.sh $H/cfg_balx_rnn50_seed$S.json ./datasets/own20_sel80_balx_mimic1000.hdf5 own100m1000_balx_rnn50_seed$S >/dev/null 2>&1 &
done; sleep 90
for T in rnn50norm rnn50lrdecay balx_rnn50 ; do ev own100m1000_${T}_seed101; done
ev own100m1000_balx_rnn50_seed111
for T in postfilt_rnn50 relrnn50 rnn100 rnn50len800 rnn50noise randctrl_rnn50; do ev own100m1000_${T}_seed101; done
for T in rnn50_noisyeval rnn50norm rnn50lrdecay postfilt_rnn50 relrnn50 rnn100 rnn50len800 rnn50noise; do r=$(rate own100m1000_${T}_seed101); [ -n "$r" ] && python -c "import sys;sys.exit(0 if $r>0.55 else 1)" && ev own100m1000_${T}_seed111; done
echo "queue13 done $(date)"
