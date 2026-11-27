#!/bin/bash

N_CARS=128
N_STEPS=600
N_EPOCHS=100
LR=1e-4
DEVICE="cpu"

# Stage 1: Base training on procedural tracks
#TRACKS=("square" "triangle" "square_narrow")
#LOAD=""
#SAVE="checkpoints/base.pth"

# Stage 2: Fine-tune on real tracks (uncomment when ready)
TRACKS=("redbull_ring" "us_track")
LOAD="checkpoints/real_tracks_best.pth"
SAVE="checkpoints/real_tracks.pth"

python3 -m train.train \
    --n_cars "$N_CARS" \
    --n_steps "$N_STEPS" \
    --n_epochs "$N_EPOCHS" \
    --lr "$LR" \
    --device "$DEVICE" \
    --tracks "${TRACKS[@]}" \
    --load "$LOAD" \
    --save "$SAVE"