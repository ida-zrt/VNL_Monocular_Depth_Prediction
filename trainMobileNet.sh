#!/bin/bash
#SBATCH -J VNL_Train
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH -t 30:00:00
#SBATCH --gres=gpu:1
module load anaconda3/2019.07
source activate VNL
python -u train_nyu_metric.py --dataroot ./datasets/sjtuRoad \
                                --dataset road \
                                --cfg_file lib/configs/mobilenet_v2_sjtuRoad_class \
                                --epoch 50
                                # --resume \
