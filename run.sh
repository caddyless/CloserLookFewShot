#!/usr/bin/env bash
cd /home/lijin/project/CloserLookFewShot
source activate py36

python train.py --dataset miniImagenet --method densenet --model ResNet10 --stop_epoch 400
python train.py --dataset miniImagenet --method densenet --model ResNet10 --stop_epoch 400