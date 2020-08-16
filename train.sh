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
                                --cfg_file lib/configs/resnext101_32x4d_sjtuRoad_class.yaml \
                                --load_ckpt ../VNL_Weight_Pretrained/kitti_eigen.pth \
                                --resume \
                                --epoch 80
