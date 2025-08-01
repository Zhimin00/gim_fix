# -*- coding: utf-8 -*-
# @Author  : xuelun

import glob
import torch
import imagesize
import torch.nn.functional as F


from os.path import join

from torch.utils.data import Dataset

class GL3DDataset(Dataset):
    def __init__(self,
                 root_dir,          # data root dit
                 npz_root,          # data info, like, overlap, image_path, depth_path
                 seq_name,          # current sequence
                 mode,              # train or val or test
                 min_overlap_score,
                 max_overlap_score,
                 max_resize,        # max edge after resize
                 df,                # general is 8 for ResNet w/ pre 3-layers
                 padding,           # padding image for batch training
                 augment_fn,        # augmentation function
                 max_samples,       # max sample in current sequence
                
                 **kwargs):
        super().__init__()

        self.root = join(root_dir, seq_name)

        paths = glob.glob(join(self.root, '*.txt'))

        lines = []
        for path in paths:
            with open(path, 'r') as file:
                scene_id = path.rpartition('/')[-1].rpartition('.')[0].split('_')[0]
                line = file.readline().strip().split()
                lines.append([scene_id] + line)

        self.pairs = sorted(lines)

        self.df = df
        self.max_resize = max_resize
        self.padding = padding

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]

        scene_id = pair[0]

        img_name0 = pair[1].rpartition('.')[0]
        img_name1 = pair[2].rpartition('.')[0]

        img_path0 = join(self.root, '{}_{}.png'.format(scene_id, img_name0))
        img_path1 = join(self.root, '{}_{}.png'.format(scene_id, img_name1))

        K0 = torch.tensor(list(map(float, pair[5:14])), dtype=torch.float).reshape(3, 3)
        K1 = torch.tensor(list(map(float, pair[14:23])), dtype=torch.float).reshape(3, 3)

        # read image size

        T_0to1 = torch.tensor(list(map(float, pair[23:])), dtype=torch.float).reshape(4, 4)
        
        # image_names = [img_path0, img_path1]
        # images, camera_intrinsics = load_images_with_intrinsics(image_names, intrinsics=[K0_ori, K1_ori])
        # K0, K1 = camera_intrinsics
        # print(K0.shape)
        # print(images[0]['img'].shape)
        data = {
            'img_path0': img_path0,
            'img_path1': img_path1,
            # image transform
            'T_0to1': T_0to1,  # (4, 4)
            'K0': K0,  # (3, 3)
            'K1': K1,
            # pair information
            'dataset_name': 'GL3D',
            'scene_id': scene_id,
            'pair_id': f'{idx}-{idx}',
            'pair_names': (img_name0,
                           img_name1),
            'covisible0': float(pair[3]),
            'covisible1': float(pair[4]),
        }

        return data