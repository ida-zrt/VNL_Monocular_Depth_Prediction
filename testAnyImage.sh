#!/bin/bash
#SBATCH -J VNL_Train
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=log.%j.out
#SBATCH --error=log.%j.err
#SBATCH -t 30:00:00
#SBATCH --gres=gpu:2
# "./lib/configs/resnext101_32x4d_sjtuRoad_class.yaml"
# "./outputs/Aug16-22-37-14_node3/ckpt/final.pth"

module load anaconda3/2019.07
source activate VNL
# python  ./test_any_images.py \
# 		--dataroot    ./ \
# 		--dataset     any \
# 		--cfg_file     lib/configs/resnext101_32x4d_sjtuRoad_class \
# 		--load_ckpt   ../VNL_Weight_Pretrained/kitti_eigen.pth

# python  ./test_any_images.py \
# 		--dataroot    ./ \
# 		--dataset     any \
# 		--cfg_file     lib/configs/resnext101_32x4d_sjtuRoad_class \
# 		--load_ckpt   ./outputs/Aug16-22-37-14_node3/ckpt/final.pth

python  ./test_any_images.py \
		--dataroot    ./ \
		--dataset     any \
		--cfg_file     lib/configs/mobilenet_v2_sjtuRoad_class \
		--load_ckpt   ./outputs/Aug20-02-39-59_node1/ckpt/final.pth
