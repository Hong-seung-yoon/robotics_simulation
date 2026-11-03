#!/bin/bash
# failrec seed101 끝나면 failrec 중단 → 후보 120개 annotate (sim 동시 1개)
cd ~/IsaacLab_RS; source ~/anaconda3/etc/profile.d/conda.sh; conda activate lerobot-arena; export PYTHONUNBUFFERED=1; L=logs/hw2
until grep -q "^== ta20own50_trim10_seed101" $L/failrec_own50.out 2>/dev/null; do sleep 10; done
sleep 15; pkill -9 -f "hw2_scripts/failrec.sh"; pkill -9 -f "failrec_ta20own50_trim10_seed111"; sleep 5
echo "failrec stopped after seed101 $(date)"
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 \
  --auto --input_file datasets/cand120_raw.hdf5 --output_file datasets/cand120_raw_annotated.hdf5 --headless > $L/cand120_raw_annotate.log 2>&1
grep "Exported [0-9]* (out of" $L/cand120_raw_annotate.log; echo "done $(date)"
