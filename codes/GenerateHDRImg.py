# encoding=utf-8

import os
import cv2
import numpy as np

import scripts.data_io as io


def gen_hdr(img_path, align_ratio_path, save_path):
    """
    :param img_path:
    :param align_ratio_path:
    :param save_path:
    :return:
    """
    img, align_ratio = io.imread_uint16_png(img_path, align_ratio_path)
    print('Align ratio: {:.5f}.'.format(align_ratio))

    img = np.clip(np.tanh(img), 0, 1) * 255.0
    img = img.round().astype(np.uint8)
    cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    print('{:s} saved.'.format(save_path))


if __name__ == '__main__':
    root = '../results/HDRUNet_tanh_l1_bs_16_ps_160/000_Valid_SingleFrame_FirstStage'
    # root = '../results/my'

    abs_path = os.path.abspath(root)
    if not os.path.isdir(root):
        print('[Err]: invalid root.')
        exit(-1)

    parent_root =  os.path.abspath(os.path.join(root, ".."))
    res_dir = parent_root  + '/hdr'
    if not os.path.isdir(res_dir):
        os.makedirs(res_dir)

    ext = '.png'
    img_path_list = [root + '/' + x for x in os.listdir(root) if x.endswith(ext)]
    alra_path_list =  [root + '/' + x for x in os.listdir(root) if x.endswith('.npy')]

    assert len(img_path_list) == len(alra_path_list)

    for img_path in img_path_list:
        img_name = os.path.split(img_path)[-1]
        alra_name = img_name[:-4] + '_alignratio.npy'
        alra_path = root + '/' + alra_name
        if not os.path.isfile(alra_path):
            print('[Warning]: alignratio file do not exist.')
            continue

        hdr_path = res_dir + '/' + img_name[:-4] + '_hdr.png'
        gen_hdr(img_path, alra_path, hdr_path)

    print('Done.')
