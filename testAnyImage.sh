# "./lib/configs/resnext101_32x4d_sjtuRoad_class.yaml"
# "./outputs/Aug16-22-37-14_node3/ckpt/final.pth"
python  ./test_any_images.py \
		--dataroot    ./ \
		--dataset     any \
		--cfg_file     lib/configs/resnext101_32x4d_sjtuRoad_class \
		--load_ckpt   ../VNL_Weight_Pretrained/kitti_eigen.pth

python  ./test_any_images.py \
		--dataroot    ./ \
		--dataset     any \
		--cfg_file     lib/configs/resnext101_32x4d_sjtuRoad_class \
		--load_ckpt   ./outputs/Aug16-22-37-14_node3/ckpt/final.pth
