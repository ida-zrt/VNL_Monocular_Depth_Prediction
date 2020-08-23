#!/bin/bash
#SBATCH -J VNL_Train
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH -t 30:00:00
#SBATCH --gres=gpu:2
module load anaconda3/2019.07
source activate VNL
# echo "*************************************************"
# echo "Performance of model pre-trained on KITTI dataset"
# echo "*************************************************"

# python -u test_nyu_metric.py --dataroot ./datasets/sjtuRoad \
#                             --dataset road \
#                             --cfg_file lib/configs/resnext101_32x4d_sjtuRoad_class \
#                             --load_ckpt ../VNL_Weight_Pretrained/kitti_eigen.pth

echo "*************************************************"
echo "Performance of model tuned with sjtuRoad dataset"
echo "*************************************************"

python -u test_nyu_metric.py --dataroot ./datasets/sjtuRoad \
                            --dataset road \
                            --cfg_file lib/configs/resnext101_32x4d_sjtuRoad_class \
                            --load_ckpt ./outputs/Aug23-10-21-13_node1/ckpt/final.pth
