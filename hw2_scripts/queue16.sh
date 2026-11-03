#!/bin/bash
# 남은 평가(11/3 18:00 KST 이후로는 새 평가를 시작하지 않음)
cd ~/IsaacLab_RS; source ~/anaconda3/etc/profile.d/conda.sh; conda activate lerobot-arena; export PYTHONUNBUFFERED=1
L=logs/hw2; H=~/IsaacLab_RS_hw2/hw2_scripts; B=logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0; DEAD=$(date -d "2026-11-03 18:00:00" +%s)
rate(){ grep -a 'Success rate' $L/failrec_$1_eval42.log 2>/dev/null | tail -1 | awk '{print $3}'; }
ev(){ [ $(date +%s) -lt $DEAD ] || { echo "skip $1 (deadline)"; return; }; [ -n "$(rate $1)" ] && return; echo "eval $1 $(date)"; $H/failrec.sh $(ls -d $B/$1/*/models | tail -1)/model_epoch_2000.pth; [ -n "$(rate $1)" ] || echo "FAILED $1"; }
P=own100m1000
ev ${P}_rnn50lrdecay_seed101; ev ${P}_rnn50norm_seed101; ev ${P}_postcond_rnn50_seed101
if [ -n "$(rate ${P}_postcond_rnn50_seed101)" ]; then HW2_POSTURE_FLAG=0.0 ev ${P}_postcond_flag0_seed101; fi
for T in rnn50lrdecay rnn50norm; do r=$(rate ${P}_${T}_seed101); [ -n "$r" ] && python -c "import sys;sys.exit(0 if $r>0.55 else 1)" && ev ${P}_${T}_seed111; done
for T in relrnn50 rnn100 rnn50len800 rnn50noise; do ev ${P}_${T}_seed101; done
echo "queue16 done $(date)"
