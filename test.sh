#!/bin/bash
source activate torch
CUDA_VISIBLE_DEVICES=1 python test.py --name label2city --which_epoch 190
