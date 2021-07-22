#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 semi_train.py --split_num=3 --epochs=5 --M=3 --alpha=2 --mask=1.1 --loss_type=svfc --shuffle=open --mining_type=intersect --noise_rate=0.4 --batch_size=512 --train_id=1028semi_1
