#! /bin/bash
gpus=$1
weight=$2
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests         ICLNUIM --img_size 512 --DATA_ROOT /cis/home/zshao14/datasets/zeb  
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests        SceneNet --img_size 512 --DATA_ROOT /cis/home/zshao14/datasets/zeb  
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests            GL3D --img_size 512 --DATA_ROOT /cis/home/zshao14/datasets/zeb  
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests          GTASfM --img_size 512 --DATA_ROOT /cis/home/zshao14/datasets/zeb  
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests        MultiFoV --img_size 512 --DATA_ROOT /cis/home/zshao14/datasets/zeb 
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests           KITTI --img_size 512 --DATA_ROOT /cis/home/zshao14/datasets/zeb  
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests          ETH3DO --img_size 512 --DATA_ROOT /cis/home/zshao14/datasets/zeb  

python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests   RobotcarNight --img_size 512 --DATA_ROOT /cis/home/zshao14/datasets/zeb  
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests  RobotcarSeason --img_size 512 --DATA_ROOT /cis/home/zshao14/datasets/zeb  
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests RobotcarWeather --img_size 512 --DATA_ROOT /cis/home/zshao14/datasets/zeb  
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests          ETH3DI --img_size 512 --DATA_ROOT /cis/home/zshao14/datasets/zeb  
python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests      BlendedMVS --img_size 512 --DATA_ROOT /cis/home/zshao14/datasets/zeb  

