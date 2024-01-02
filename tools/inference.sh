#!/usr/bin/env bash
python demo/demo_vid.py configs/vid/temporal_roi_align/selsa_troialign_faster_rcnn_r50_dc5_7e_SUIT.py \
    --input "../data/SUIT/demo/input/KIM HS 64F 159 62__01914451.mp4" \
    --checkpoint work_dirs/selsa_troialign_faster_rcnn_r50_dc5_7e_SUIT/latest.pth \
    --output "../data/SUIT/demo/output/KIM HS 64F 159 62__01914451.mp4"