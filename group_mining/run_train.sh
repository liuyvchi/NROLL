#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 train.py --M=4 --alpha=3 --mask=1.1 --loss_type=svfc --shuffle=open --mining_type=intersect --batch_size=512 --noise_rate=0.5 --train_id=1025_1
