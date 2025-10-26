# 每个mask区域均是不同的embedding进行处理
# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""
import torch
import einops
import numpy as np
from torch import nn
from collections import OrderedDict
import torch.nn.functional as F

from clip import clip
from clip.model import LayerNorm
from .prompt_templates import FLIP_real_templates, FLIP_manipulate_templates, lowfreq_templates, midfreq_templates, highfreq_templates
from .base_module.freqformer import FreqFormer


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
    def __init__(self, in_dim = 512, mlp_ratio = 4.0, out_dim = 256):
        super().__init__()
        hidden_dim = int(in_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, out_dim)
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

    
class FullFreqPerception(nn.Module):
    def __init__(self):
        super().__init__()

        self.freq_proj = nn.Linear(192, 512)
        
        self.attn1 = Transformer(width = 512, layers = 2, heads = 8)
        self.img_attn = Transformer(width = 512, layers = 2, heads = 8)
        self.text_attn = Transformer(width = 512, layers = 2, heads = 8)
        
    def forward(self, img_feat, freq_feat, difffreq_patch_feat, text_feat):
        '''
        img_feat: [N, 197, 512]
        freq_feat: [N, num_rings, 192]
        text_feat: [2, 512]
        '''

        freq_feat = self.freq_proj(freq_feat)
        
        patch_feat = img_feat[:, 1: ]
        patch_feat = self.attn1(patch_feat, freq_feat, freq_feat)

        patch_feat = torch.cat([patch_feat, difffreq_patch_feat], dim = 1)

        cls_feat = img_feat[:, 0].unsqueeze(1)
        text_feat = einops.repeat(text_feat, 'n c -> repeat n c', repeat = freq_feat.shape[0])
        
        refine_img_feat = self.img_attn(cls_feat, patch_feat, patch_feat)

        refine_text_feat = self.text_attn(text_feat, patch_feat, patch_feat)
        
        return refine_img_feat, refine_text_feat
        
    
class DiffFreqPerception(nn.Module):
    def __init__(self):
        super().__init__() 

        self.freq_proj = nn.Linear(192, 512)
        self.attn = Transformer(width = 512, layers = 2, heads = 8)  
        self.img_attn = Transformer(width = 512, layers = 2, heads = 8)

    def forward(self, img_feat, lowfreq_feat, midfreq_feat, highfreq_feat):
        lowfreq_feat = self.freq_proj(lowfreq_feat)
        midfreq_feat = self.freq_proj(midfreq_feat)
        highfreq_feat = self.freq_proj(highfreq_feat)
        
        patch_feat = img_feat[:, 1: ]
        cls_feat = img_feat[:, 0].unsqueeze(1)
        
        # low img feat
        low_patch_feat = self.attn(patch_feat, lowfreq_feat, lowfreq_feat)
        low_cls_feat = self.img_attn(cls_feat, low_patch_feat, low_patch_feat)
        #

        # mid img feat
        mid_patch_feat = self.attn(patch_feat, midfreq_feat, midfreq_feat)
        mid_cls_feat = self.img_attn(cls_feat, mid_patch_feat, mid_patch_feat)
        #

        # high img feat
        high_patch_feat = self.attn(patch_feat, highfreq_feat, highfreq_feat)
        high_cls_feat = self.img_attn(cls_feat, high_patch_feat, high_patch_feat)
        #
        freq_cls_feat = torch.cat([low_cls_feat, mid_cls_feat, high_cls_feat], dim = 1)
        
        return freq_cls_feat
    
             
            
class FreqNext(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.model, _ = clip.load(cfg['model']['backbone'], device = self.cfg['device'])
        self.freqformer = FreqFormer(num_rings = self.cfg['model']['num_rings'], max_pool = self.cfg['model']['max_pool'], \
                                     embed_dim = 192, depth = 12, num_heads = 3, \
                                    lowfreq_range = cfg['model']['low_freq_range'], middlefreq_range = cfg['model']['middle_freq_range'])
        self.diff_freq_perception = DiffFreqPerception()
        self.full_freq_perception = FullFreqPerception()

        self.projection_head = ProjHead(in_dim = 512, mlp_ratio = 4.0, out_dim = 256)


    def forward(self, input_dict):
        img = input_dict['img']
        aug1_img = input_dict['aug1_img']
        aug2_img = input_dict['aug2_img']
        labels = input_dict['label']
        freq = input_dict['freq']

        ### image branch
        img_feat = self.model.visual.extract_visual_feature(img) # N * 197 * 768
        img_feat = img_feat @ self.model.visual.proj
        ### 

        ### freq branch
        lowfreq_feat, midfreq_feat, highfreq_feat = self.freqformer(freq) # N * num_rings * 192
        ###

        ### text freq brunch
        tokenized_low_prompts = clip.tokenize(lowfreq_templates).cuda(non_blocking = True, device = self.cfg['device'])
        low_prompts_feat = self.model.encode_text(tokenized_low_prompts)
        
        tokenized_mid_prompts = clip.tokenize(midfreq_templates).cuda(non_blocking = True, device = self.cfg['device'])
        middle_prompts_feat = self.model.encode_text(tokenized_mid_prompts)

        tokenized_high_prompts = clip.tokenize(highfreq_templates).cuda(non_blocking = True, device = self.cfg['device'])
        high_prompts_feat = self.model.encode_text(tokenized_high_prompts)

        mean_lowfreq_prompts_feat = low_prompts_feat.mean(dim = 0)
        mean_midfreq_prompts_feat = middle_prompts_feat.mean(dim = 0)
        mean_highfreq_prompts_feat = high_prompts_feat.mean(dim = 0)

        ensemble_freq_prompt_feat = torch.stack([mean_lowfreq_prompts_feat, mean_midfreq_prompts_feat, mean_highfreq_prompts_feat], dim = 0)
        ###

        ### Different Frequency cls feat
        freq_cls_feat = self.diff_freq_perception(img_feat, lowfreq_feat, midfreq_feat, highfreq_feat)

        ### text cls brunch
        tokenized_real_prompts = clip.tokenize(FLIP_real_templates).cuda(non_blocking = True, device = self.cfg['device'])
        real_prompts_feat = self.model.encode_text(tokenized_real_prompts)

        tokenized_spoof_prompts = clip.tokenize(FLIP_manipulate_templates).cuda(non_blocking = True, device = self.cfg['device'])
        spoof_prompts_feat = self.model.encode_text(tokenized_spoof_prompts)

        mean_real_prompts_feat = real_prompts_feat.mean(dim = 0)
        mean_spoof_prompts_feat = spoof_prompts_feat.mean(dim = 0)

        ensemble_prompt_feat = torch.stack([mean_real_prompts_feat, mean_spoof_prompts_feat], dim = 0)
        ###
        
        ### cls loss
        img_feat, ensemble_prompt_feat = self.full_freq_perception(img_feat, torch.cat([lowfreq_feat, midfreq_feat, highfreq_feat], dim = 1), freq_cls_feat, ensemble_prompt_feat)
        ensemble_prompt_feat = ensemble_prompt_feat / ensemble_prompt_feat.norm(dim = -1, keepdim = True)
        img_feat = img_feat / img_feat.norm(dim = -1, keepdim = True)


        logit_scale = self.model.logit_scale.exp()
        logits = logit_scale * torch.matmul(img_feat, ensemble_prompt_feat.transpose(-1, -2))
        logits = logits.squeeze(1)
        ### 
        
        ### SimCLR loss
        aug1_feat = self.model.encode_image(aug1_img)
        aug1_feat = aug1_feat / aug1_feat.norm(dim = -1, keepdim = True)
        out_aug1_feat = self.projection_head(aug1_feat)

        aug2_feat = self.model.encode_image(aug2_img)
        aug2_feat = aug2_feat / aug2_feat.norm(dim = -1, keepdim = True)
        out_aug2_feat = self.projection_head(aug2_feat)
        ###

        ### MSE loss
        text_embedding_v1 = []
        text_embedding_v2 = []

        for label in labels:

            if label == 1:
                available_indices = np.arange(0, len(FLIP_manipulate_templates))
                pair_1 = np.random.choice(available_indices, len(FLIP_manipulate_templates) // 2)
                pair_2 = np.setdiff1d(available_indices, pair_1)

                spoof_texts_v1 = [spoof_prompts_feat[i] for i in pair_1]
                spoof_texts_v2 = [spoof_prompts_feat[i] for i in pair_2]

                spoof_texts_v1 = torch.stack(spoof_texts_v1, dim = 0)
                spoof_texts_v2 = torch.stack(spoof_texts_v2, dim = 0)

                text_embedding_v1.append(spoof_texts_v1.mean(dim = 0))
                text_embedding_v2.append(spoof_texts_v2.mean(dim = 0))
            
            elif label == 0:
                available_indices = np.arange(0, len(FLIP_real_templates))
                pair_1 = np.random.choice(available_indices, len(FLIP_real_templates) // 2)
                pair_2 = np.setdiff1d(available_indices, pair_1)

                real_texts_v1 = [real_prompts_feat[i] for i in pair_1]
                real_texts_v2 = [real_prompts_feat[i] for i in pair_2]

                real_texts_v1 = torch.stack(real_texts_v1, dim = 0)
                real_texts_v2 = torch.stack(real_texts_v2, dim = 0)

                text_embedding_v1.append(real_texts_v1.mean(dim = 0))
                text_embedding_v2.append(real_texts_v2.mean(dim = 0))

        text_embed_v1 = torch.stack(text_embedding_v1, dim = 0)
        text_embed_v2 = torch.stack(text_embedding_v2, dim = 0)


        text_embed_v1_norm = text_embed_v1 / text_embed_v1.norm(dim = -1, keepdim = True)
        text_embed_v2_norm = text_embed_v2 / text_embed_v2.norm(dim = -1, keepdim = True)

        aug1_text_dot_product = F.cosine_similarity(aug1_feat, text_embed_v1_norm)
        aug2_text_dot_product = F.cosine_similarity(aug2_feat, text_embed_v2_norm)
        ###

        output_dict = {}
        output_dict['logits'] = logits
        output_dict['aug1_feat'] = out_aug1_feat
        output_dict['aug2_feat'] = out_aug2_feat
        output_dict['aug1_text_dot'] = aug1_text_dot_product
        output_dict['aug2_text_dot'] = aug2_text_dot_product
        output_dict['lowfreq'] = lowfreq_feat
        output_dict['midfreq'] = midfreq_feat
        output_dict['highfreq'] = highfreq_feat
        output_dict['diff_freq_cls'] = freq_cls_feat
        output_dict['diff_freq_text'] = ensemble_freq_prompt_feat

        return output_dict




