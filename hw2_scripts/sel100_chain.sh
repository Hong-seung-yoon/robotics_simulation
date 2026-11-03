#!/bin/bash
cd ~/IsaacLab_RS; source ~/anaconda3/etc/profile.d/conda.sh; conda activate lerobot-arena; L=logs/hw2
until grep -q "^done" $L/annotate_cand.out 2>/dev/null; do sleep 20; done
python -I ~/IsaacLab_RS_hw2/hw2_scripts/select100.py || exit 1
python -I ~/IsaacLab_RS_hw2/summarize_demos.py datasets/own20_sel80_raw_annotated.hdf5 | sed -n 2p
~/IsaacLab_RS_hw2/hw2_scripts/pipeline.sh datasets/own20_sel80_raw_annotated.hdf5 own20_sel80_raw 1000 > $L/pipeline_own20_sel80_raw.out 2>&1
