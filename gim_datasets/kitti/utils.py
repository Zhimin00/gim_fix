# -*- coding: utf-8 -*-
# @Author  : xuelun

import cv2
import math
import torch

import numpy as np

from PIL import Image

from gim_datasets.utils import imread_color, get_resized_wh, get_divisible_wh


def pad_bottom_right(inp, pad_size, ret_mask=False):
    h = pad_size[0]
    h = math.ceil(h / 8) * 8
    pad_size = (h, pad_size[1])
    # assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size[0], pad_size[1]), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
    elif inp.ndim == 3:
        padded = np.zeros((pad_size[0], pad_size[1], inp.shape[-1]), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
    else:
        raise NotImplementedError()

    if ret_mask:
        mask = np.zeros((pad_size[0], pad_size[1]), dtype=bool)
        mask[:inp.shape[0], :inp.shape[1]] = True

    return padded, mask


def read_depth(path):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(path), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(float) / 256.
    depth[depth_png == 0] = -1.

    padded = np.zeros((400, 1300), dtype=depth.dtype)
    padded[:depth.shape[0], :depth.shape[1]] = depth

    return padded


def read_images(path, max_resize, df, padding, augment_fn=None, image=None):
    """
    Args:
        path: string
        max_resize (int): max image size after resied
        df (int, optional): image size division factor.
                            NOTE: this will change the final image size after img_resize
        padding (bool): If set to 'True', zero-pad resized images to squared size.
        augment_fn (callable, optional): augments images with pre-defined visual effects
        image: RGB image
    Returns:
        image (torch.tensor): (1, h, w)
        mask (torch.tensor): (h, w)
        scale (torch.tensor): [w/w_new, h/h_new]
    """
    # read image
    assert max_resize is not None

    image = imread_color(path, augment_fn) if image is None else image # (w,h,3) image is RGB
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # resize image
    w, h = image.shape[1], image.shape[0]
    if max(w, h) > max_resize:
        w_new, h_new = get_resized_wh(w, h, max_resize) # make max(w, h) to max_size
    else:
        w_new, h_new = w, h

    # w_new, h_new = get_divisible_wh(w_new, h_new, df) # make image divided by df and must <= max_size
    image = cv2.resize(image, (w_new, h_new))  # (w',h',3)
    gray = cv2.resize(gray, (w_new, h_new))  # (w',h',3)
    scale = torch.tensor([w / w_new, h / h_new], dtype=torch.float)

    # padding
    mask = None
    if padding:
        image, _ = pad_bottom_right(image, (int(max_resize/3.25), max_resize), ret_mask=False)
        gray, mask = pad_bottom_right(gray, (int(max_resize/3.25), max_resize), ret_mask=True)
        mask = torch.from_numpy(mask)

    gray = torch.from_numpy(gray).float()[None] / 255 # (1,h,w)
    image = torch.from_numpy(image).float() / 255  # (h,w,3)
    image = image.permute(2,0,1) # (3,h,w)

    resize = [h_new, w_new]

    return gray, image, scale, resize, mask
