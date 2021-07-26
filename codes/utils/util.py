# encoding=utf-8

import logging
import os
import random
from collections import OrderedDict
from datetime import datetime

import cv2
import math
import numpy as np
import torch
import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def OrderedYaml():
    '''yaml orderedDict support'''
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


####################
# miscellaneous
####################


def get_timestamp():
    """
    :return:
    """
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    """
    :param path:
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    """
    :param paths:
    :return:
    """
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    """
    :param path:
    :return:
    """
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))

        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))

        os.rename(path, new_name)
        print('{:s} renamed to {:s}.'.format(path, new_name))

    os.makedirs(path)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    elif n_dim == 2:
        # img_np = tensor.numpy()
        img_np = tensor.numpy()
        img_np = np.expand_dims(img_np, axis=2)
        img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
        img_np = np.squeeze(img_np)
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
    elif out_type == np.uint16:
        img_np = (img_np * 65535.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def tensor2numpy(tensor):
    """
    :param tensor:
    :return:
    """
    img_np = tensor.numpy()
    img_np[img_np < 0] = 0
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    return img_np.astype(np.float32)


def cvt2uint16(img, tone_mapping=False):
    """
    :param img:
    :param tone_mapping:
    :return:
    """
    ## ----- compute align_ratio
    img_max_val = img.max()
    print('IMG max val: ', img_max_val, end=' ')

    align_ratio = (2 ** 16 - 1) / img_max_val  # 255 / max_val
    print("align ratio: ", align_ratio)

    ## tone mapping
    if tone_mapping:
        img = np.clip(np.tanh(img), 0, 1) * align_ratio
    else:
        img = img * align_ratio

    img = np.round(img).astype(np.uint16)

    return img


def cvt2uint8(img, tone_mapping=False):
    """
    :param img:
    :param tone_mapping:
    :return:
    """
    ## ----- compute align_ratio
    img_max_val = img.max()
    print('IMG max val: ', img_max_val, end=' ')

    align_ratio = float(2 ** 8 - 1) / img_max_val  # 255 / max_val
    print("align ratio: ", align_ratio)

    ## tone mapping
    if tone_mapping:
        img = np.clip(np.tanh(img), 0, 1) * align_ratio
    else:
        img = img * align_ratio

    img = np.round(img).astype(np.uint8)

    return img

    # uint8_img = np.round(img * align_ratio).astype(np.uint8)
    # return uint8_img


def save_img_uint8(img_path, img):
    """
    :param img_path:
    :param img:
    :return:
    """
    ## ----- compute align_ratio
    img_max_val = img.max()
    print('IMG max val: ', img_max_val, end=' ')

    align_ratio = (2 ** 8 - 1) / img_max_val  # 255 / max_val
    print("align ratio: ", align_ratio)

    uint8_image_gt = np.round(img * align_ratio).astype(np.uint8)

    cv2.imwrite(img_path, uint8_image_gt)
    print("{:s} saved.".format(img_path))

    return None


def save_img_with_ratio_uin8(img_path, alignratio_path, img):
    """
    :param img_path:
    :param alignratio_path:
    :param img:
    :return:
    """
    ## ----- compute align_ratio
    img_max = img.max()
    print("image max: {:.3f}".format(img_max), end=" ")

    align_ratio = (2 ** 8 - 1) / img_max  # 255 / max_val
    print("align ratio: {:.3f}".format(align_ratio))

    ## save align ratio
    np.save(alignratio_path, align_ratio)

    ## multipy align-ratio, rounding and convert to uint8
    uint8_image_gt = np.round(img * align_ratio).astype(np.uint8)

    cv2.imwrite(img_path, uint8_image_gt)
    print("{:s} saved.".format(img_path))

    return None


def save_img_with_ratio_uint16(img_path, alignratio_path, img):
    """
    :param img_path:
    :param alignratio_path:
    :param img:
    :return:
    """
    ## ----- compute align_ratio
    img_max = img.max()
    print("image max: {:.3f}".format(img_max), end=" ")

    align_ratio = (2 ** 16 - 1) / img_max  # 65535 / max_val
    print("align ratio: {:.3f}".format(align_ratio))

    ## save align ratio
    np.save(alignratio_path, align_ratio)

    ## multipy align-ratio, rounding and convert to uint16
    uint16_image_gt = np.round(img * align_ratio).astype(np.uint16)

    cv2.imwrite(img_path, uint16_image_gt)
    print("{:s} saved.".format(img_path))

    return None


def generate_paths(dir, name, ext=".png"):
    """
    :param dir:
    :param name:
    :return:
    """
    id = name[:-len(ext)]
    img_path = os.path.join(dir, id + ext)
    alignratio_path = os.path.join(dir, id + '_alignratio.npy')
    return img_path, alignratio_path


def save_img(img, img_path, mode='RGB'):
    """
    :param img:
    :param img_path:
    :param mode:
    :return:
    """
    cv2.imwrite(img_path, img)


def save_npy(img, img_path):
    """
    :param img:
    :param img_path:
    :return:
    """
    img = np.squeeze(img)
    np.save(img_path, img)


def calculate_psnr(img1, img2):
    """
    :param img1:
    :param img2:
    :return:
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    # return 20 * math.log10(255.0 / math.sqrt(mse))
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calculate_normalized_psnr(img1, img2, norm):
    """
    :param img1:
    :param img2:
    :param norm:
    :return:
    """
    normalized_psnr = -10 * np.log10(np.mean(np.power(img1 / norm - img2 / norm, 2)))
    if normalized_psnr == 0:
        return float('inf')
    return normalized_psnr


def mu_tonemap(hdr_image, mu=5000):
    return np.log(1 + mu * hdr_image) / np.log(1 + mu)


def tanh_norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    bounded_hdr = np.tanh(hdr_image / norm_value)
    return mu_tonemap(bounded_hdr, mu)


def calculate_tonemapped_psnr(res, ref, percentile=99, gamma=2.24):
    res = res ** gamma
    ref = ref ** gamma
    norm_perc = np.percentile(ref, percentile)
    tonemapped_psnr = -10 * np.log10(
        np.mean(np.power(tanh_norm_mu_tonemap(ref, norm_perc) - tanh_norm_mu_tonemap(res, norm_perc), 2)))
    return tonemapped_psnr
