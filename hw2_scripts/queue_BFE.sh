#!/bin/bash
# sim 순서: (진행 중 failrec 종료 대기) → B: 주인18 + Mimic192 → F: 주인45(50중 annotate 통과) + Mimic1000 → E1: 조교20 + Mimic1000
cd ~/IsaacLab_RS; L=logs/hw2; H=~/IsaacLab_RS_hw2/hw2_scripts
while pgrep -f "hw2_scripts/failrec.sh" >/dev/null; do sleep 30; done
echo "B start $(date)"; $H/pipeline.sh datasets/own18_raw_m192_annotated.hdf5 own18_raw_m192 192 > $L/pipeline_own18_raw_m192.out 2>&1; tail -2 $L/pipeline_own18_raw_m192.out
echo "F start $(date)"; $H/pipeline.sh datasets/own45_raw_m1000_annotated.hdf5 own45_raw_m1000 1000 > $L/pipeline_own45_raw_m1000.out 2>&1; tail -2 $L/pipeline_own45_raw_m1000.out
echo "E1 start $(date)"; $H/pipeline.sh datasets/ta20_raw_m1000_annotated.hdf5 ta20_raw_m1000 1000 > $L/pipeline_ta20_raw_m1000.out 2>&1; tail -2 $L/pipeline_ta20_raw_m1000.out
