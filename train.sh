#!/bin/bash
source activate torch
#python train.py --name label2city_512p --continue_train
#python train.py --name label2city_512p
CUDA_VISIBLE_DEVICES=0 python train.py --continue_train

