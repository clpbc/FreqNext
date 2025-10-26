# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""
# base
from .vit_cls import ViT_cls
from .vit_mean import ViT_mean
from .vit_mean_dr import ViT_mean_dr
from .navit import NaViT

# from .vit_mean import ViT_mean
from .clip import CLIP

# from .resnet import ResNet
# from .resnet_highfreq import ResNet_HighFreq
from .xception import Xception
from .xception_groupnorm import Xception_groupnorm
from .xception_instancenorm import Xception_instancenorm
# FAS (Face Anti-Spoofing)
from .flip import FLIP

# from .me_fas import MEFAS

# FFD (Face Forgery Detection)
from .F3Net import F3Net
from .recce import Recce
from .freqnext import FreqNext


# from .vit_cls import ViT
# from .F3Net import F3Net
# # from .simplefreq import SimpleFreq
# from .wavevit import wavevit_s
# from .basefreq2 import BaseFreq2
# from .basefreq3 import BaseFreq3
# from .basefreq4 import BaseFreq4

# from .freqnext import FreqNext



modelActions = {
    # "resnet": ResNet,
    # "resnet_highfreq": ResNet_HighFreq,
    "vit_cls": ViT_cls,
    "vit_mean": ViT_mean,
    "vit_mean_dr": ViT_mean_dr,
    "navit": NaViT,
    "clip": CLIP,
    "flip": FLIP,
    "freqnext": FreqNext,
    "xception": Xception,
    "xception_groupnorm": Xception_groupnorm,
    "xception_instancenorm": Xception_instancenorm,
    # "me_fas": MEFAS,
    "recce": Recce,
    "f3net": F3Net,
    # 'vit': ViT,
    # 'flip': FLIP,
    # 'f3net': F3Net,
    # 'freqnet': freqnet,
    # # 'simplefreq': SimpleFreq,
    # # 'wavevit': wavevit_s,
    # # 'basefreq2': BaseFreq2,
    # # 'basefreq3': BaseFreq3,
}


def BuildModel(cfg, log):

    if cfg["model"]["mode"] in modelActions:
        model = modelActions[cfg["model"]["mode"]](cfg)
        log.write(f"Loading Model: {type(model).__name__}")

    else:
        raise ValueError("model mode is not supported")

    return model
