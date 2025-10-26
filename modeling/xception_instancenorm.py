# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""

from torch import nn
import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from .base_module.xception_instancenorm import Xception_backbone


pretrained_path = r'./pretrained/xception_df40_fsall.pth'


class Xception_instancenorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg['model']
        
        num_classes = self.cfg['num_classes']
        mode = self.cfg['model_mode']
        inc = self.cfg['inc']
        dropout = self.cfg['dropout']
        self.pretrained = True

        self.model = Xception_backbone(num_classes = num_classes, mode = mode, inc = inc, dropout = dropout)
        
        if self.pretrained:
            state_dict = torch.load(pretrained_path)
            cleaned_state_dict = self.clean_state_dict(state_dict)
            consume_prefix_in_state_dict_if_present(cleaned_state_dict, 'module.backbone.')
            
            self.model.load_state_dict(state_dict, strict = False)
            
    def clean_state_dict(self, state_dict):
        # 创建新的state_dict，移除所有running_mean和running_var
        new_state_dict = {}
        for k, v in state_dict.items():
            if 'running_mean' not in k and 'running_var' not in k:
                new_state_dict[k] = v
        return new_state_dict
        

    def forward(self, input_dict):
        img = input_dict['img']

        logits, feat = self.model(img)

        output_dict = {}
        output_dict['logits'] = logits

        return output_dict
        