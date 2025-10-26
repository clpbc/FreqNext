# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

# Official Losses
from torch.nn.modules.loss import *


from .infonceloss import InfoNCELoss
from .ortholoss import OrthoLoss
from .contraloss import ContraLoss

# RECCE Losses
from .recce_loss import RecceReconsLoss, RecceMLLoss
