# encoding=utf-8

'''create dataset and dataloader'''
import logging

import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
    """
    :param dataset:
    :param dataset_opt:
    :param opt:
    :param sampler:
    :return:
    """
    phase = dataset_opt['phase']
    if phase == 'train':
        if opt['dist']:
            world_size = torch.distributed.get_world_size()
            num_workers = dataset_opt['n_workers']
            assert dataset_opt['batch_size'] % world_size == 0
            batch_size = dataset_opt['batch_size'] // world_size
            shuffle = False
        else:
            if dataset_opt['debug']:
                num_workers = 0
            else:
                num_workers = dataset_opt['n_workers'] * len(opt['gpu_ids'])

            batch_size = dataset_opt['batch_size']
            shuffle = True

        print('Number of workers: {:d}.'.format(num_workers))

        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           num_workers=num_workers,
                                           sampler=sampler,
                                           drop_last=True,
                                           pin_memory=False)
    else:
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=1,
                                           shuffle=False,
                                           num_workers=1,
                                           pin_memory=False)


def create_dataset(dataset_opt):
    """
    :param dataset_opt:
    :return:
    """
    mode = dataset_opt['mode']

    if mode == 'LQ_condition':
        from codes.data.LQ_condition_dataset import LQ_Dataset as D
    elif mode == 'LQGT_condition':
        from codes.data.LQGT_condition_dataset import LQGT_dataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))

    dataset = D(dataset_opt)

    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           dataset_opt['name']))
    return dataset
