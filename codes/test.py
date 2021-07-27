# encoding=utf-8

import argparse
import logging
import os
import os.path as osp
import time
from collections import OrderedDict

import cv2
import numpy as np

import codes.options.options as option
import codes.utils.util as util
from codes.data import create_dataset, create_dataloader
from codes.models import create_model

#### options
parser = argparse.ArgumentParser()

parser.add_argument('-opt',
                    type=str,
                    default='./options/test/test_HDRUNet.yml',
                    help='Path to options YMAL file.')

opt = option.parse(parser.parse_args().opt, is_train=False)
opt = option.dict_to_nonedict(opt)

util.mkdirs((path for key, path in opt['path'].items()
             if not key == 'experiments_root' and 'pretrain_model' not in key and 'resume' not in key))
util.setup_logger('base', opt['path']['log'], 'test_' + opt['name'],
                  level=logging.INFO,
                  screen=True,
                  tofile=True)
logger = logging.getLogger('base')
logger.info(option.dict2str(opt))

#### Create test dataset and dataloader
test_loaders = []
for phase, dataset_opt in sorted(opt['datasets'].items()):
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)
    logger.info('Number of test images in [{:s}]: {:d}'.format(dataset_opt['name'], len(test_set)))
    test_loaders.append(test_loader)

model = create_model(opt)
for test_loader in test_loaders:
    test_set_name = test_loader.dataset.opt['name']
    logger.info('\nTesting [{:s}]...'.format(test_set_name))
    test_start_time = time.time()
    dataset_dir = osp.join(opt['path']['results_root'], test_set_name)
    util.mkdir(dataset_dir)

    test_results = OrderedDict()
    test_results['psnr'] = []

    save_ext = ".jpg"
    for i, data in enumerate(test_loader):
        ## Need GT or not
        need_GT = False if test_loader.dataset.opt['dataroot_GT'] is None else True

        ## ----- Get image path and image name
        img_path = data['GT_path'][0] if need_GT else data['LQ_path'][0]
        img_name = os.path.split(img_path)[-1]
        print("Processing {:s}...".format(img_name))

        ## ----- Set up save path
        # save_img_path, alignratio_path = util.generate_paths(dataset_dir, img_name, save_ext)

        ## ----- inference
        model.feed_data(data, need_GT=need_GT)
        model.test()
        out_dict = model.get_current_visuals(need_GT=need_GT)

        ## ------ Post processing
        hdr = util.tensor2numpy(out_dict['SR'])  # dtype: float32 BGR
        ## -----

        ## ----- Read in LDR image
        ldr = cv2.imread(data["LQ_path"][0], cv2.IMREAD_COLOR)
        h, w, c = ldr.shape

        ## ----- convert to uint8 format
        # hdr_ = util.cvt2uint8(hdr, tone_mapping=False)
        hdr_ = np.clip(np.round(hdr * 255.0 + 0.5), 0, 255).astype(np.uint8)

        result = np.zeros((h * 2, w, c), dtype=np.uint8)
        result[:h, :, :] = ldr
        result[h:2 * h, :, :] = hdr_

        save_path = dataset_dir + '/' + img_name[:-len(save_ext)] + save_ext
        cv2.imwrite(save_path, result)
        print("{:s} saved.".format(save_path))

        ## -----logging
        logger.info('{:20s} tested | {:d}/{:d}\n'.format(img_name, i + 1, len(test_loader)))


## ----- Save output image
# util.save_img_with_ratio_uint16(save_img_path, alignratio_path, hdr)
# util.save_img_with_ratio_uin8(save_img_path, alignratio_path, hdr)
# util.save_img_uint8(save_img_path, hdr)  # hdr