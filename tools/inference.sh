#!/usr/bin/env bash

for i in {40..50}
do
python demo/demo_vid.py configs/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_SUIT.py \
    --input "../data/SUIT/demo/input/CNUH_DC04_BPB1_00${i}.mp4" \
    --checkpoint work_dirs/selsa_troialign_faster_rcnn_r50_dc5_7e_SUIT/latest.pth \
    --output "../data/SUIT/demo/output/CNUH_DC04_BPB1_00${i}.mp4"
done