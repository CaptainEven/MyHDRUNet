# encoding=utf-8

import os
import os.path
import sys
from multiprocessing import Pool

import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from progress_bar import ProgressBar


def extract_sub():
    """
    A multii-thread tool to crop sub imags.
    :return:
    """
    input_dir = '/mnt/diskc/even/Ldr2HdrData/medium'
    output_dir = '/mnt/diskc/even/Ldr2HdrData/medium_sub'

    n_threads = 8  # number of threads: 1, 8, 10
    crop_sz = 480  # crop size
    step = 240  # crop stride
    thres_sz = 48
    compress_level = 0  # 3 is the default value in cv2
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print('mkdir [{:s}] ...'.format(output_dir))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(output_dir))
        # sys.exit(1)

    img_list = []
    for root, _, file_list in sorted(os.walk(input_dir)):
        path = [os.path.join(root, x) for x in file_list]  # assume only images in the input_folder
        img_list.extend(path)
        # for file_name in file_list:
        #    if os.path.splitext(file_name)[1] == '.png':
        #        img_list.append(os.path.join(root, file_name))
    print("Total {:d} images.".format(len(img_list)))

    name = os.path.split(img_list[0])[-1]
    ext = "." + name.split('.')[-1]
    print("Ext: ", ext)

    def update(arg):
        """
        :param arg:
        :return:
        """
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(n_threads)
    for path in img_list:
        pool.apply_async(worker,
                         args=(path, output_dir, crop_sz, step, thres_sz, compress_level, ext),
                         callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')


def worker(path, save_dir, crop_sz, step, thres_sz, compression_level, ext=".png"):
    """
    :param path:
    :param save_dir:
    :param crop_sz:
    :param step:
    :param thres_sz:
    :param compression_level:
    :return:
    """
    img_name = os.path.basename(path)
    # img_name = '_'.join(path.split('/')[-4:])

    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:  # Gray image
        h, w = img.shape
    elif n_channels == 3:  # BGR or RGB image
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)

    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    idx = 0
    for x in h_space:
        for y in w_space:
            idx += 1

            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]

            crop_img = np.ascontiguousarray(crop_img)
            # var = np.var(crop_img / 255)
            # if var > 0.008:
            #     print(img_name, index_str, var)

            save_name = img_name.replace(ext, '_s{:03d}{:s}'.format(idx, ext))
            save_path = save_dir + '/' + save_name
            cv2.imwrite(save_path,
                        crop_img,
                        [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
            print('{:s} saved.'.format(save_path))

    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    extract_sub()
