#! /bin/bash
gpus=$1
weight=$2
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests         ICLNUIM  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path /cis/home/zshao14/checkpoints/spider_warp_0727/checkpoint-best.pth 
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests        SceneNet  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path /cis/home/zshao14/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests            GL3D  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path /cis/home/zshao14/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests          GTASfM  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path /cis/home/zshao14/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests        MultiFoV  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path /cis/home/zshao14/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests   RobotcarNight  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path /cis/home/zshao14/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests  RobotcarSeason  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path /cis/home/zshao14/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests RobotcarWeather  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path /cis/home/zshao14/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests           KITTI  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path /cis/home/zshao14/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests          ETH3DI  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path /cis/home/zshao14/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests          ETH3DO  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path /cis/home/zshao14/checkpoints/spider_warp_0727/checkpoint-best.pth
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests      BlendedMVS  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path /cis/home/zshao14/checkpoints/spider_warp_0727/checkpoint-best.pth

