#!/bin/bash
# sim 순서(E1 보류): F 종료 대기 → F s101 분류 → ① s101/s111 평가(+기록·분류·obs 검증) → ② s101 → V1 s101 → B-trim → 나머지 분류
cd ~/IsaacLab_RS; L=logs/hw2; H=~/IsaacLab_RS_hw2/hw2_scripts; B=logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
ck(){ d=$(ls -d $B/$1/*/models 2>/dev/null | tail -1); [ -n "$d" ] && echo $d/model_epoch_$2.pth; }
waittrain(){ while pgrep -f "train_cfg.py.*--name $1" >/dev/null; do sleep 60; done; }
while kill -0 $1 2>/dev/null; do sleep 30; done; echo "F done $(date)"; grep -E "seed1[01]1:|ABORT" $L/pipeline_own45_raw_m1000.out
$H/failrec.sh $(ck own45_raw_m1000_seed101 2000)
for S in 101 111; do waittrain own100m1000_relobs_seed$S; echo "relobs s$S eval $(date)"; $H/failrec.sh $(ck own100m1000_relobs_seed$S 2000)
  python3 -I $H/check_relobs.py $L/failrec_own100m1000_relobs_seed${S}_eval42; done
waittrain own100m1000_rnn50_seed101; echo "rnn50 eval $(date)"; $H/failrec.sh $(ck own100m1000_rnn50_seed101 2000)
waittrain own100m1000_rnn30h1000_e4000_seed101; echo "V1 eval $(date)"; $H/failrec.sh $(ck own100m1000_rnn30h1000_e4000_seed101 4000)
echo "Btrim start $(date)"; $H/pipeline.sh datasets/own15_trim10_m192_annotated.hdf5 own15_trim10_m192 192 > $L/pipeline_own15_trim10_m192.out 2>&1; grep -E "seed1[01]1:|ABORT" $L/pipeline_own15_trim10_m192.out
for S in 101 111; do for T in own45_raw_m1000 own18_raw_m192 own100_nomimic own15_trim10_m192; do
  C=$(ck ${T}_seed$S 2000); [ -n "$C" ] && [ -e "$C" ] && [ ! -e $L/failrec_${T}_seed${S}_eval42.csv ] && $H/failrec.sh $C; done; done
echo "all done $(date)"
