#! /bin/bash
gpus=$1
weight=$2
size1=$3
size2=$4
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests         ICLNUIM --img_size $size1 --fine_size $size2 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path  ~/shared/zshao14/checkpoints/spider_mast3r_warp_0730/checkpoint-best.pth 
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests        SceneNet --img_size $size1 --fine_size $size2 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path  ~/shared/zshao14/checkpoints/spider_mast3r_warp_0730/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests            GL3D --img_size $size1 --fine_size $size2 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path  ~/shared/zshao14/checkpoints/spider_mast3r_warp_0730/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests          GTASfM --img_size $size1 --fine_size $size2 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path  ~/shared/zshao14/checkpoints/spider_mast3r_warp_0730/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests        MultiFoV --img_size $size1 --fine_size $size2 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path  ~/shared/zshao14/checkpoints/spider_mast3r_warp_0730/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests   RobotcarNight --img_size $size1 --fine_size $size2 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path  ~/shared/zshao14/checkpoints/spider_mast3r_warp_0730/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests  RobotcarSeason --img_size $size1 --fine_size $size2 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path  ~/shared/zshao14/checkpoints/spider_mast3r_warp_0730/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests RobotcarWeather --img_size $size1 --fine_size $size2 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path  ~/shared/zshao14/checkpoints/spider_mast3r_warp_0730/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests           KITTI --img_size $size1 --fine_size $size2 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path  ~/shared/zshao14/checkpoints/spider_mast3r_warp_0730/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests          ETH3DI --img_size $size1 --fine_size $size2 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path  ~/shared/zshao14/checkpoints/spider_mast3r_warp_0730/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests          ETH3DO --img_size $size1 --fine_size $size2 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path  ~/shared/zshao14/checkpoints/spider_mast3r_warp_0730/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests      BlendedMVS --img_size $size1 --fine_size $size2 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path  ~/shared/zshao14/checkpoints/spider_mast3r_warp_0730/checkpoint-best.pth


