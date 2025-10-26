# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""

from .base_module.resnet_highfreq import *
from torch import nn

resnet_model = {
    'resnet34': resnet34,
    'resnet50': resnet50,
}

class ResNet_HighFreq(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.model = resnet_model[cfg['model']['backbone']](num_classes = 2, method = cfg['model']['method'])

    def forward(self, input_dict):
        img = input_dict['img']

        logits = self.model(img)

        output_dict = {}
        output_dict['logits'] = logits

        return output_dict
        