#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 baseline.py --loss_type=svfc --batch_size=512 --noise_rate=0.3 --train_id=0925_2
