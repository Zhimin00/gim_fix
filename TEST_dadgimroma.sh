#! /bin/bash
# gpus=$1
# weight=$2
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests           KITTI  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path  /cis/home/zshao14/checkpoints/spiderfm_conv_ms_0822/checkpoint-best.pth --outdir_name spiderfm_conv_ms_0822-512
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests         ICLNUIM  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path  /cis/home/zshao14/checkpoints/spiderfm_conv_ms_0822/checkpoint-best.pth --outdir_name spiderfm_conv_ms_0822-512
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests        SceneNet  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path  /cis/home/zshao14/checkpoints/spiderfm_conv_ms_0822/checkpoint-best.pth --outdir_name spiderfm_conv_ms_0822-512
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests            GL3D  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path  /cis/home/zshao14/checkpoints/spiderfm_conv_ms_0822/checkpoint-best.pth --outdir_name spiderfm_conv_ms_0822-512
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests          GTASfM  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path  /cis/home/zshao14/checkpoints/spiderfm_conv_ms_0822/checkpoint-best.pth --outdir_name spiderfm_conv_ms_0822-512
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests        MultiFoV  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path  /cis/home/zshao14/checkpoints/spiderfm_conv_ms_0822/checkpoint-best.pth --outdir_name spiderfm_conv_ms_0822-512
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests   RobotcarNight  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path  /cis/home/zshao14/checkpoints/spiderfm_conv_ms_0822/checkpoint-best.pth --outdir_name spiderfm_conv_ms_0822-512
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests  RobotcarSeason  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path  /cis/home/zshao14/checkpoints/spiderfm_conv_ms_0822/checkpoint-best.pth --outdir_name spiderfm_conv_ms_0822-512
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests RobotcarWeather  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path  /cis/home/zshao14/checkpoints/spiderfm_conv_ms_0822/checkpoint-best.pth --outdir_name spiderfm_conv_ms_0822-512
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests          ETH3DI  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path  /cis/home/zshao14/checkpoints/spiderfm_conv_ms_0822/checkpoint-best.pth --outdir_name spiderfm_conv_ms_0822-512
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests          ETH3DO  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path  /cis/home/zshao14/checkpoints/spiderfm_conv_ms_0822/checkpoint-best.pth --outdir_name spiderfm_conv_ms_0822-512
# python test.py --gpus $gpus --weight $weight --version 100h --test --batch_size 1 --tests      BlendedMVS  --DATA_ROOT /cis/home/zshao14/datasets/zeb --checkpoint_path  /cis/home/zshao14/checkpoints/spiderfm_conv_ms_0822/checkpoint-best.pth --outdir_name spiderfm_conv_ms_0822-512

# python analysis.py --dir dump/zeb/spiderfm_conv_ms_0822-512 --wid spiderfm --version 100h --verbose
gpus=$1
python test.py --gpus $gpus --weight gim_roma --version 100h --test --batch_size 1 --tests            GL3D --DATA_ROOT /cis/home/zshao14/datasets/zeb --outdir_name spgimroma
python test.py --gpus $gpus --weight gim_roma --version 100h --test --batch_size 1 --tests           KITTI --img_size 1240 --DATA_ROOT /cis/home/zshao14/datasets/zeb --outdir_name spgimroma
python test.py --gpus $gpus --weight gim_roma --version 100h --test --batch_size 1 --tests          GTASfM --DATA_ROOT /cis/home/zshao14/datasets/zeb --outdir_name spgimroma
python test.py --gpus $gpus --weight gim_roma --version 100h --test --batch_size 1 --tests         ICLNUIM --DATA_ROOT /cis/home/zshao14/datasets/zeb --outdir_name spgimroma
python test.py --gpus $gpus --weight gim_roma --version 100h --test --batch_size 1 --tests        MultiFoV --DATA_ROOT /cis/home/zshao14/datasets/zeb --outdir_name spgimroma
python test.py --gpus $gpus --weight gim_roma --version 100h --test --batch_size 1 --tests      BlendedMVS --DATA_ROOT /cis/home/zshao14/datasets/zeb --outdir_name spgimroma
python test.py --gpus $gpus --weight gim_roma --version 100h --test --batch_size 1 --tests        SceneNet --DATA_ROOT /cis/home/zshao14/datasets/zeb --outdir_name spgimroma
python test.py --gpus $gpus --weight gim_roma --version 100h --test --batch_size 1 --tests          ETH3DI --img_size 1600 --DATA_ROOT /cis/home/zshao14/datasets/zeb --outdir_name spgimroma
python test.py --gpus $gpus --weight gim_roma --version 100h --test --batch_size 1 --tests          ETH3DO --img_size 1600 --DATA_ROOT /cis/home/zshao14/datasets/zeb --outdir_name spgimroma
python test.py --gpus $gpus --weight gim_roma --version 100h --test --batch_size 1 --tests   RobotcarNight --DATA_ROOT /cis/home/zshao14/datasets/zeb --outdir_name spgimroma
python test.py --gpus $gpus --weight gim_roma --version 100h --test --batch_size 1 --tests  RobotcarSeason --max_samples 2000 --DATA_ROOT /cis/home/zshao14/datasets/zeb --outdir_name spgimroma
python test.py --gpus $gpus --weight gim_roma --version 100h --test --batch_size 1 --tests RobotcarWeather --DATA_ROOT /cis/home/zshao14/datasets/zeb --outdir_name spgimroma

python analysis.py --dir dump/zeb/spgimroma --wid gim_roma --version 100h --verbose

