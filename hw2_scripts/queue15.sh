#!/bin/bash
# 추가 생성 종료 대기 → 균등 x 데이터/학습 → 소스 선택 시험 → 준비된 모델부터 우선순위대로 평가(유휴 없음), 11/3 18:00 KST에 중단
cd ~/IsaacLab_RS; source ~/anaconda3/etc/profile.d/conda.sh; conda activate lerobot-arena; export PYTHONUNBUFFERED=1
L=logs/hw2; H=~/IsaacLab_RS_hw2/hw2_scripts; B=logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
DEAD=$(date -d "2026-11-03 18:00:00" +%s)
rate(){ grep -a 'Success rate' $L/failrec_$1_eval42.log 2>/dev/null | tail -1 | awk '{print $3}'; }
done_(){ [ -n "$(rate $1)" ]; }
ready(){ pgrep -f "train_cfg.py.*--name $1( |$)" >/dev/null && return 1; ls $B/$1/*/models/model_epoch_2000.pth >/dev/null 2>&1; }
ev(){ echo "eval $1 $(date)"; $H/failrec.sh $(ls -d $B/$1/*/models | tail -1)/model_epoch_2000.pth; python -I $H/xbin_success.py $L/failrec_$1_eval42; }
while kill -0 $1 2>/dev/null; do sleep 30; done
echo "gen pool2 done $(date): $(grep -ao '[0-9]*/[0-9]* ([0-9.]*%) successful demos' $L/own20_sel80_pool2_generate.log | tail -1)"
python -I $H/balance_truck_x.py datasets/own20_sel80_balx_mimic1000.hdf5 datasets/own20_sel80_balx_mimic1000_map.csv 1000 datasets/own20_sel80_raw_mimic_raw.hdf5 datasets/own20_sel80_raw_mimic_pool2.hdf5 || echo ABORT balance
for S in 101 111; do python -c "
import json;c=json.load(open('$H/cfg_rnn50_seed101.json'));c['train']['seed']=$S;json.dump(c,open('$H/cfg_balx_rnn50_seed$S.json','w'),indent=4)"
 nohup $H/train_cfg.sh $H/cfg_balx_rnn50_seed$S.json ./datasets/own20_sel80_balx_mimic1000.hdf5 own100m1000_balx_rnn50_seed$S >/dev/null 2>&1 &
done; sleep 90
echo "place-select test start $(date)"
HW2_PLACE_SELECT=nearest_neighbor_robot_distance HW2_GEN_SEED=3 ./isaaclab.sh -p $H/generate_dataset_seed.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 --num_envs 100 --generation_num_trials 150 --input_file datasets/own20_sel80_raw_annotated.hdf5 --output_file datasets/own20_sel80_placetest.hdf5 --headless > $L/own20_sel80_placetest_generate.log 2>&1
echo "place-select test done $(date): $(grep -ao '[0-9]*/[0-9]* ([0-9.]*%) successful demos' $L/own20_sel80_placetest_generate.log | tail -1)"
PRE="own100m1000"; CONTRIB="rnn50hardx rnn50approach postfilt_rnn50 randctrl_rnn50 balx_rnn50"
while [ $(date +%s) -lt $DEAD ]; do
  L1=""; for T in rnn50hardx rnn50approach postfilt_rnn50 randctrl_rnn50 balx_rnn50; do L1="$L1 ${PRE}_${T}_seed101"; done
  L1="$L1 ${PRE}_balx_rnn50_seed111 ${PRE}_postcond_rnn50_seed101 ${PRE}_rnn50norm_seed101 ${PRE}_rnn50lrdecay_seed101"
  for T in rnn50hardx rnn50approach postfilt_rnn50 postcond_rnn50 rnn50norm rnn50lrdecay; do r=$(rate ${PRE}_${T}_seed101); [ -n "$r" ] && python -c "import sys;sys.exit(0 if $r>0.55 else 1)" && L1="$L1 ${PRE}_${T}_seed111"; done
  if done_ ${PRE}_postcond_rnn50_seed101 && ! done_ ${PRE}_postcond_flag0_seed101; then
    C=$(ls -d $B/${PRE}_postcond_rnn50_seed101/*/models | tail -1)/model_epoch_2000.pth; mkdir -p $B/${PRE}_postcond_flag0_seed101/20261103/models; ln -sf "$PWD/$C" $B/${PRE}_postcond_flag0_seed101/20261103/models/model_epoch_2000.pth
    L1="$L1 FLAG0"
  fi
  for T in relrnn50 rnn100 rnn50len800 rnn50noise; do L1="$L1 ${PRE}_${T}_seed101"; done
  did=0
  for T in $L1; do
    if [ $T = FLAG0 ]; then echo "eval postcond flag0 $(date)"; HW2_POSTURE_FLAG=0.0 $H/failrec.sh $(ls -d $B/${PRE}_postcond_flag0_seed101/*/models | tail -1)/model_epoch_2000.pth; did=1; break; fi
    done_ $T && continue
    ready $T && { ev $T; did=1; break; }
  done
  [ $did = 1 ] || sleep 120
  if [ $did = 0 ]; then n=0; for T in $L1; do [ $T = FLAG0 ] || done_ $T || n=$((n+1)); done; [ $n = 0 ] && break; fi
done
echo "queue15 done $(date)"
