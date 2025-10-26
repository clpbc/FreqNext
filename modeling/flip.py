# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from clip import clip
from .prompt_templates import FLIP_real_templates, FLIP_spoof_templates


class ProjHead(nn.Module):
    def __init__(self, in_dim=512, mlp_ratio=4.0, out_dim=256):
        super().__init__()

        hidden_dim = int(in_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

        self._initialize_weights()

    def forward(self, x):
        return self.mlp(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class FLIP(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.model, _ = clip.load(cfg["model"]["backbone"], device=self.cfg["device"])

        self.projection_head = ProjHead(in_dim=512, mlp_ratio=4.0, out_dim=256)

    def forward(self, input_dict):
        img = input_dict["img"]
        aug1_img = input_dict["aug1_img"]
        aug2_img = input_dict["aug2_img"]
        labels = input_dict["label"]

        ### image branch
        img_feat = self.model.encode_image(img)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        ###

        ### text branch
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

        ### SimCLR loss
        aug1_feat = self.model.encode_image(aug1_img)
        aug1_feat = aug1_feat / aug1_feat.norm(dim=-1, keepdim=True)
        out_aug1_feat = self.projection_head(aug1_feat)

        aug2_feat = self.model.encode_image(aug2_img)
        aug2_feat = aug2_feat / aug2_feat.norm(dim=-1, keepdim=True)
        out_aug2_feat = self.projection_head(aug2_feat)
        ###

        ### MSE loss
        text_embedding_v1 = []
        text_embedding_v2 = []

        for label in labels:

            if label == 1:
                available_indices = np.arange(0, len(FLIP_spoof_templates))
                pair_1 = np.random.choice(
                    available_indices, len(FLIP_spoof_templates) // 2
                )
                pair_2 = np.setdiff1d(available_indices, pair_1)

                spoof_texts_v1 = [spoof_prompts_feat[i] for i in pair_1]
                spoof_texts_v2 = [spoof_prompts_feat[i] for i in pair_2]

                spoof_texts_v1 = torch.stack(spoof_texts_v1, dim=0)
                spoof_texts_v2 = torch.stack(spoof_texts_v2, dim=0)

                text_embedding_v1.append(spoof_texts_v1.mean(dim=0))
                text_embedding_v2.append(spoof_texts_v2.mean(dim=0))

            elif label == 0:
                available_indices = np.arange(0, len(FLIP_real_templates))
                pair_1 = np.random.choice(
                    available_indices, len(FLIP_real_templates) // 2
                )
                pair_2 = np.setdiff1d(available_indices, pair_1)

                real_texts_v1 = [real_prompts_feat[i] for i in pair_1]
                real_texts_v2 = [real_prompts_feat[i] for i in pair_2]

                real_texts_v1 = torch.stack(real_texts_v1, dim=0)
                real_texts_v2 = torch.stack(real_texts_v2, dim=0)

                text_embedding_v1.append(real_texts_v1.mean(dim=0))
                text_embedding_v2.append(real_texts_v2.mean(dim=0))

        text_embed_v1 = torch.stack(text_embedding_v1, dim=0)
        text_embed_v2 = torch.stack(text_embedding_v2, dim=0)

        text_embed_v1_norm = text_embed_v1 / text_embed_v1.norm(dim=-1, keepdim=True)
        text_embed_v2_norm = text_embed_v2 / text_embed_v2.norm(dim=-1, keepdim=True)

        aug1_text_dot_product = F.cosine_similarity(aug1_feat, text_embed_v1_norm)
        aug2_text_dot_product = F.cosine_similarity(aug2_feat, text_embed_v2_norm)
        ###

        output_dict = {}
        output_dict["logits"] = logits
        output_dict["aug1_feat"] = out_aug1_feat
        output_dict["aug2_feat"] = out_aug2_feat
        output_dict["aug1_text_dot"] = aug1_text_dot_product
        output_dict["aug2_text_dot"] = aug2_text_dot_product

        return output_dict
