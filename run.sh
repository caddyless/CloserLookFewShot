#!/usr/bin/env bash
cd /home/lijin/project/CloserLookFewShot
source activate py36

python train.py --dataset miniImagenet --method attennet --model AttenNet10 --stop_epoch 400 --k_num 30
python train.py --dataset miniImagenet --method attennet --model AttenNet10 --stop_epoch 400 --k_num 20
python train.py --dataset miniImagenet --method attennet --model AttenNet10 --stop_epoch 400 --k_num 15
python train.py --dataset miniImagenet --method attennet --model AttenNet10 --stop_epoch 400 --k_num 25
