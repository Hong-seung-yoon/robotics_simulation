#!/bin/bash
# queue2(B→noMimic→F→E1) 종료 대기 → B-trim(주인15 trim10 + Mimic192) → 실패 분류(seed101 우선, 이후 111)
cd ~/IsaacLab_RS; L=logs/hw2; H=~/IsaacLab_RS_hw2/hw2_scripts; B=logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0
while pgrep -f "hw2_scripts/queue2" >/dev/null; do sleep 60; done
echo "Btrim start $(date)"; $H/pipeline.sh datasets/own15_trim10_m192_annotated.hdf5 own15_trim10_m192 192 > $L/pipeline_own15_trim10_m192.out 2>&1; tail -2 $L/pipeline_own15_trim10_m192.out
ck(){ ls -d $B/$1/*/models 2>/dev/null | tail -1 | sed 's|$|/model_epoch_2000.pth|'; }
for S in 101 111; do
  for T in own45_raw_m1000 own18_raw_m192 own100_nomimic ta20_raw_m1000 own15_trim10_m192; do
    C=$(ck ${T}_seed$S); [ -n "$C" ] && [ -e "$C" ] && [ ! -e $L/failrec_${T}_seed${S}_eval42.csv ] && $H/failrec.sh $C
  done
done
echo "all done $(date)"
