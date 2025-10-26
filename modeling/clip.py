# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""
import torch
from torch import nn

from clip import clip
from .prompt_templates import FLIP_real_templates, FLIP_spoof_templates


class CLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.model, _ = clip.load(cfg["model"]["backbone"], device=self.cfg["device"])

    def forward(self, input_dict):
        img = input_dict["img"]

        ### image branch
        img_feat = self.model.encode_image(img)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        ### text brunch
        tokenized_real_prompts = clip.tokenize(FLIP_real_templates).cuda(
            non_blocking=True, device=self.cfg["device"]
        )
        real_prompts_feat = self.model.encode_text(tokenized_real_prompts)

        tokenized_spoof_prompts = clip.tokenize(FLIP_spoof_templates).cuda(
            non_blocking=True, device=self.cfg["device"]
        )
        spoof_prompts_feat = self.model.encode_text(tokenized_spoof_prompts)

        mean_real_prompts_feat = real_prompts_feat.mean(dim=0)
        mean_spoof_prompts_feat = spoof_prompts_feat.mean(dim=0)

        ensemble_prompt_feat = torch.stack(
            [mean_real_prompts_feat, mean_spoof_prompts_feat], dim=0
        )
        ensemble_prompt_feat = ensemble_prompt_feat / ensemble_prompt_feat.norm(
            dim=-1, keepdim=True
        )
        ###

        ### cls loss
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * img_feat @ ensemble_prompt_feat.T
        ###

        output_dict = {}
        output_dict["logits"] = logits

        return output_dict
