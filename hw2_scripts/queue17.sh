#!/bin/bash
# lrdecay 평가 종료 대기 → norm → postcond flag 1, 0 → relrnn50 → rnn100 → rnn50len800 → rnn50noise. s101 >= 0.56이면 s111 바로 평가. 새 평가 시작은 21:00 KST까지
cd ~/IsaacLab_RS; source ~/anaconda3/etc/profile.d/conda.sh; conda activate lerobot-arena; export PYTHONUNBUFFERED=1
L=logs/hw2; H=~/IsaacLab_RS_hw2/hw2_scripts; B=logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0; DEAD=$(date -d "2026-11-03 21:00:00" +%s); P=own100m1000
rate(){ grep -a 'Success rate' $L/failrec_$1_eval42.log 2>/dev/null | tail -1 | awk '{print $3}'; }
sum_(){ python -I $H/eval_summary.py $1 >> $L/results_live.txt 2>&1; }
ev(){ [ $(date +%s) -lt $DEAD ] || { echo "skip $1 (21:00)"; return; }; [ -n "$(rate $1)" ] && return; echo "eval $1 $(date)"; $H/failrec.sh $(ls -d $B/$1/*/models | tail -1)/model_epoch_2000.pth; if [ -n "$(rate $1)" ]; then sum_ $1; else echo "RESULT FAILED $1" >> $L/results_live.txt; fi; }
ev2(){ ev ${P}_$1_seed101; r=$(rate ${P}_$1_seed101); [ -n "$r" ] && python -c "import sys;sys.exit(0 if $r>=0.56 else 1)" && ev ${P}_$1_seed111; }
while kill -0 $1 2>/dev/null; do sleep 20; done; sleep 5; sum_ ${P}_rnn50lrdecay_seed101
r=$(rate ${P}_rnn50lrdecay_seed101); [ -n "$r" ] && python -c "import sys;sys.exit(0 if $r>=0.56 else 1)" && ev ${P}_rnn50lrdecay_seed111
ev2 rnn50norm; ev ${P}_postcond_rnn50_seed101
[ -n "$(rate ${P}_postcond_rnn50_seed101)" ] && HW2_POSTURE_FLAG=0.0 ev ${P}_postcond_flag0_seed101
r=$(rate ${P}_postcond_rnn50_seed101); [ -n "$r" ] && python -c "import sys;sys.exit(0 if $r>=0.56 else 1)" && ev ${P}_postcond_rnn50_seed111
for T in relrnn50 rnn100 rnn50len800 rnn50noise; do ev2 $T; done
echo "queue17 done $(date)"
