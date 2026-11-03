#!/bin/bash
# noisyeval s101(진행 중) 대기 → 변형 s101 전부 → seed101 상위 3개(>=50%) s111 → 최종 후보 체크포인트 비교(epoch 1000/1500, 양 seed)
cd ~/IsaacLab_RS; source ~/anaconda3/etc/profile.d/conda.sh; conda activate lerobot-arena; export PYTHONUNBUFFERED=1
L=logs/hw2; H=~/IsaacLab_RS_hw2/hw2_scripts; B=logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
ev(){ while pgrep -f "train_cfg.py.*--name $1( |$)" >/dev/null; do sleep 60; done; echo "eval $1 $(date)"; $H/failrec.sh $(ls -d $B/$1/*/models | tail -1)/model_epoch_2000.pth; }
rate(){ grep -a 'Success rate' $L/failrec_$1_eval42.log 2>/dev/null | tail -1 | awk '{print $3}'; }
while kill -0 $1 2>/dev/null; do sleep 30; done
for V in rnn50norm rnn50lrdecay relrnn50 rnn50noise rnn50len800 rnn100; do ev own100m1000_${V}_seed101; done
TOP=$(for V in rnn50_noisyeval rnn50norm rnn50lrdecay relrnn50 rnn50noise rnn50len800 rnn100; do echo "$(rate own100m1000_${V}_seed101) $V"; done | awk '$1>=0.50' | sort -rn | head -3 | awk '{print $2}')
echo "top by seed101: $TOP"
for V in $TOP; do ev own100m1000_${V}_seed111; done
BEST=$(for V in rnn50 $TOP; do a=$(rate own100m1000_${V}_seed101); b=$(rate own100m1000_${V}_seed111); echo "$(python3 -c "print(($a+$b)/2)") $V"; done | sort -rn | head -1 | awk '{print $2}')
echo "best mean: $BEST"
for S in 101 111; do for E in 1500 1000; do
  C=$(ls -d $B/own100m1000_${BEST}_seed$S/*/models | tail -1)/model_epoch_$E.pth
  echo "ckpt eval $BEST s$S e$E $(date)"
  ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --checkpoint $C --headless --seed 42 --num_rollouts 100 --horizon 800 > $L/ckpt_${BEST}_s${S}_e${E}_eval42.log 2>&1
  echo "ckpt $BEST s$S e$E: $(grep -a 'Success rate' $L/ckpt_${BEST}_s${S}_e${E}_eval42.log | tail -1)"
done; done
echo "queue10 done $(date)"
