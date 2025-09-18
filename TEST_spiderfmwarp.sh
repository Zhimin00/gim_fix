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
datasets=(KITTI ICLNUIM SceneNet GL3D GTASfM BlendedMVS MultiFoV RobotcarNight RobotcarSeason RobotcarWeather ETH3DI ETH3DO)

for ds in "${datasets[@]}"; do
    python test.py --gpus 8 --weight spiderfmwarp --version 100h --test --batch_size 1 \
        --tests $ds \
        --DATA_ROOT /cis/home/zshao14/datasets/zeb \
        --outdir_name spiderfmwarp-512 \
        --img_size 512
done
python analysis.py --dir dump/zeb/spiderfmwarp-512 --wid spiderfmwarp --version 100h --verbose

