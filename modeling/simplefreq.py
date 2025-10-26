# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""
import timm
import torch
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from torch import nn

from clip import clip
from clip.model import LayerNorm
from .prompt_templates import FLIP_real_templates, FLIP_spoof_templates


class QuickGELU(nn.Module):
		def forward(self, x: torch.Tensor):
				return x * torch.sigmoid(1.702 * x)

                
class ResidualAttentionBlock(nn.Module):
		def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
				super().__init__()

				self.attn = nn.MultiheadAttention(d_model, n_head, batch_first = True)
				self.ln_q = LayerNorm(d_model)
				self.mlp = nn.Sequential(OrderedDict([
						("c_fc", nn.Linear(d_model, d_model * 4)),
						("gelu", QuickGELU()),
						("c_proj", nn.Linear(d_model * 4, d_model))
				]))
				self.ln_2 = LayerNorm(d_model)
				self.attn_mask = attn_mask

		def attention(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
				self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
				return self.attn(x, y, z, need_weights=False, attn_mask=self.attn_mask)[0]

		def forward(self, input: list):
				x, y, z = input
				x = x + self.attention(self.ln_q(x), self.ln_q(y), self.ln_q(z))
				x = x + self.mlp(self.ln_2(x))
				return [x, input[1], input[2]]

class Transformer(nn.Module):
		def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
				super().__init__()
				self.width = width
				self.layers = layers
				self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

		def forward(self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
				return self.resblocks([x, y, z])[0]


class ProjHead(nn.Module):
    def __init__(self, in_dim = 512, mlp_dim = 4096, out_dim = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace = True),
            nn.Linear(mlp_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace = True),
            nn.Linear(mlp_dim, out_dim)
        )

        self._initialize_weights()

    def forward(self, x):
        return self.mlp(x)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SimpleFreq(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.model, _ = clip.load(cfg['model']['backbone'], device = self.cfg['device'])
        self.freq_extracter = timm.create_model('efficientnet_b0', pretrained = True, in_chans = 1, num_classes = 768)

        self.freq_attn = Transformer(width = 768, layers = 2, heads = 8)

        self.projection_head = ProjHead(in_dim = 512, mlp_dim = 4096, out_dim = 256)


    def forward(self, aug_img, aug1_img, aug2_img, freq, labels):
        ### image brunch
        # img_feat = self.model.encode_image(aug_img)
        # img_feat /= img_feat.norm(dim = -1, keepdim = True)
        img_feat = self.model.visual.extract_visual_feature(aug_img) # N * 197 * 768

        freq_feat = self.freq_extracter(freq.unsqueeze(1)).unsqueeze(1)  # N * 768

        img_feat = self.freq_attn(img_feat, freq_feat, freq_feat)
        img_feat = img_feat[:, 0, :] @ self.model.visual.proj
        img_feat /= img_feat.norm(dim = -1, keepdim = True)
        ###

        ### text brunch
        tokenized_real_prompts = clip.tokenize(FLIP_real_templates).cuda(non_blocking = True, device = self.cfg['device'])
        real_prompts_feat = self.model.encode_text(tokenized_real_prompts)

        tokenized_spoof_prompts = clip.tokenize(FLIP_spoof_templates).cuda(non_blocking = True, device = self.cfg['device'])
        spoof_prompts_feat = self.model.encode_text(tokenized_spoof_prompts)

        mean_real_prompts_feat = real_prompts_feat.mean(dim = 0)
        mean_spoof_prompts_feat = spoof_prompts_feat.mean(dim = 0)

        ensemble_prompt_feat = torch.stack([mean_real_prompts_feat, mean_spoof_prompts_feat], dim = 0)
        ensemble_prompt_feat /= ensemble_prompt_feat.norm(dim = -1, keepdim = True)
        ###
        
        ### cls loss
        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * img_feat @ ensemble_prompt_feat.T
        ### 

        # ### SimCLR loss
        # aug1_feat = self.model.encode_image(aug1_img)
        # aug1_feat /= aug1_feat.norm(dim = -1, keepdim = True)
        # out_aug1_feat = self.projection_head(aug1_feat)

        # aug2_feat = self.model.encode_image(aug2_img)
        # aug2_feat /= aug2_feat.norm(dim = -1, keepdim = True)
        # out_aug2_feat = self.projection_head(aug2_feat)
        # ###

        # ### MSE loss
        # text_embedding_v1 = []
        # text_embedding_v2 = []

        # for label in labels:

        #     if label == 1:
        #         available_indices = np.arange(0, len(FLIP_spoof_templates))
        #         pair_1 = np.random.choice(available_indices, len(FLIP_spoof_templates) // 2)
        #         pair_2 = np.setdiff1d(available_indices, pair_1)

        #         spoof_texts_v1 = [spoof_prompts_feat[i] for i in pair_1]
        #         spoof_texts_v2 = [spoof_prompts_feat[i] for i in pair_2]

        #         spoof_texts_v1 = torch.stack(spoof_texts_v1, dim = 0)
        #         spoof_texts_v2 = torch.stack(spoof_texts_v2, dim = 0)

        #         text_embedding_v1.append(spoof_texts_v1.mean(dim = 0))
        #         text_embedding_v2.append(spoof_texts_v2.mean(dim = 0))
            
        #     elif label == 0:
        #         available_indices = np.arange(0, len(FLIP_real_templates))
        #         pair_1 = np.random.choice(available_indices, len(FLIP_real_templates) // 2)
        #         pair_2 = np.setdiff1d(available_indices, pair_1)

        #         real_texts_v1 = [real_prompts_feat[i] for i in pair_1]
        #         real_texts_v2 = [real_prompts_feat[i] for i in pair_2]

        #         real_texts_v1 = torch.stack(real_texts_v1, dim = 0)
        #         real_texts_v2 = torch.stack(real_texts_v2, dim = 0)

        #         text_embedding_v1.append(real_texts_v1.mean(dim = 0))
        #         text_embedding_v2.append(real_texts_v2.mean(dim = 0))

        # text_embed_v1 = torch.stack(text_embedding_v1, dim = 0)
        # text_embed_v2 = torch.stack(text_embedding_v2, dim = 0)


        # text_embed_v1_norm = text_embed_v1 / text_embed_v1.norm(dim = -1, keepdim = True)
        # text_embed_v2_norm = text_embed_v2 / text_embed_v2.norm(dim = -1, keepdim = True)

        # aug1_text_dot_product = F.cosine_similarity(aug1_feat, text_embed_v1_norm)
        # aug2_text_dot_product = F.cosine_similarity(aug2_feat, text_embed_v2_norm)
        # ### 

        # return logits, out_aug1_feat, out_aug2_feat, aug1_text_dot_product, aug2_text_dot_product
        return logits




