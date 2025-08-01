#! /bin/bash
gpus=$1
weight=$2
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests         ICLNUIM --fine_size 1600 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path ~/workspace/checkpoints/spider_warp_0727/checkpoint-best.pth 
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests        SceneNet --fine_size 1600 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path ~/workspace/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests            GL3D --fine_size 1600 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path ~/workspace/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests          GTASfM --fine_size 1600 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path ~/workspace/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests        MultiFoV --fine_size 1600 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path ~/workspace/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests   RobotcarNight --fine_size 1600 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path ~/workspace/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests  RobotcarSeason --fine_size 1600 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path ~/workspace/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests RobotcarWeather --fine_size 1600 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path ~/workspace/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests           KITTI --fine_size 1600 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path ~/workspace/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests          ETH3DI --fine_size 1600 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path ~/workspace/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests          ETH3DO --fine_size 1600 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path ~/workspace/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests      BlendedMVS --fine_size 1600 --DATA_ROOT ~/shared/zshao14/zeb --checkpoint_path ~/workspace/checkpoints/spider_warp_0727/checkpoint-best.pth

