# encoding=utf-8

import torch
import logging

import codes.models.modules.UNet_arch as UNet_arch
logger = logging.getLogger('base')


####################
# define network
####################
#### Generator
def define_G(opt):
    """
    :param opt:
    :return:
    """
    opt_net = opt['network_G']
    which_model = opt_net['which_model_G']

    if which_model == 'HDRUNet':
        netG = UNet_arch.HDRUNet(in_nc=opt_net['in_nc'],
                                 out_nc=opt_net['out_nc'],
                                 nf=opt_net['nf'],
                                 act_type=opt_net['act_type'])
    else:
        raise NotImplementedError('Generator model [{:s}] not recognized'.format(which_model))

    return netG