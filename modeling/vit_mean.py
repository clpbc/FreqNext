# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""

from .base_module.VisionTransformer_mean import VisionTransformer
from torch import nn



class ViT_mean(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        
        self.model = VisionTransformer(img_size = cfg['model']['img_size'], patch_size = cfg['model']['patch_size'], \
                                       in_channels = cfg['model']['in_channels'], num_classes = cfg['model']['num_classes'], \
                                        embed_dim = cfg['model']['embed_dim'], num_heads = cfg['model']['num_heads'], \
                                        mlp_ratio = cfg['model']['mlp_ratio'], dropout = cfg['model']['dropout'], \
                                        drop_path_rate = cfg['model']['drop_path_rate'])


    def forward(self, input_dict):
        img = input_dict['img']
        
        logits = self.model(img)

        output_dict = {}
        output_dict['logits'] = logits

        return output_dict




