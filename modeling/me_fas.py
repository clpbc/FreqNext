# encoding: utf-8
"""
@author:  clpbc
@contact: clpszdnb@gmail.com
"""

import math
import os
from typing import Dict

import torch
import torch.nn.functional as F
from torch import nn

from clip import clip
from clip.model import CLIP


class text2image_attention_transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 768)  # Adjust the dimension from 512 to 768
        r = 4
        self.ln = nn.LayerNorm(768)
        self.mlp = nn.Sequential(
            nn.Linear(768, 768 // r),
            nn.ReLU(),
            nn.Linear(768 // r, 768),
        )

    def forward(self, vision_prompt, text_prompt):
        mean_vision_prompt = torch.mean(vision_prompt, dim=1)
        mean_text_prompt = torch.mean(text_prompt, dim=1)

        transformed_prompt = self.fc(mean_text_prompt)
        affinities = F.softmax(
            torch.einsum("nc, mc -> nm", mean_vision_prompt, transformed_prompt)
            / math.sqrt(transformed_prompt.shape[-1]),
            -1,
        )
        aug_image_feat = self.mlp(
            self.ln(torch.einsum("nm, mc -> nc", affinities, transformed_prompt))
        )
        return aug_image_feat.unsqueeze(1).expand(-1, vision_prompt.shape[1], -1)


class image2text_attention_transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 512)  # Adjust the dimension from 768 to 512
        r = 4
        self.ln = nn.LayerNorm(512)
        self.mlp = nn.Sequential(
            nn.Linear(512, 512 // r),
            nn.ReLU(),
            nn.Linear(512 // r, 512),
        )

    def forward(self, text_prompt, vision_prompt):
        mean_vision_prompt = torch.mean(vision_prompt, dim=1)
        mean_text_prompt = torch.mean(text_prompt, dim=1)

        transformed_prompt = self.fc(mean_vision_prompt)
        affinities = F.softmax(
            torch.einsum("nc,mc->nm", mean_text_prompt, transformed_prompt)
            / math.sqrt(transformed_prompt.shape[-1]),
            -1,
        )
        aug_text_feat = self.mlp(
            self.ln(torch.einsum("nm,mc->nc", affinities, transformed_prompt))
        )
        return aug_text_feat.unsqueeze(1).expand(-1, text_prompt.shape[1], -1)


class CLIP_MEFAS(CLIP):
    def __init__(self, clip_config):
        super().__init__(
            embed_dim=clip_config["embed_dim"],
            image_resolution=clip_config["image_resolution"],
            vision_layers=clip_config["vision_layers"],
            vision_width=clip_config["vision_width"],
            vision_patch_size=clip_config["vision_patch_size"],
            context_length=clip_config["context_length"],
            vocab_size=clip_config["vocab_size"],
            transformer_heads=clip_config["transformer_heads"],
            transformer_layers=clip_config["transformer_layers"],
            transformer_width=clip_config["transformer_width"],
        )

        self.nctx = clip_config["nctx"]
        self.prompt_depth = clip_config["prompt_depth"]

        self.mask_mode = clip_config["mask_mode"]
        self.mask_ratio = clip_config["mask_ratio"]
        self.mask_depth = clip_config["mask_depth"]

        transformer_width = clip_config["transformer_width"]
        vision_width = clip_config["vision_width"]

        self.image_projections_prompt = nn.ModuleList(
            [image2text_attention_transform() for _ in range(self.prompt_depth - 1)]
        )
        self.text_projection_image = nn.Parameter(
            torch.empty(transformer_width, vision_width)
        )

    def forward(
        self, image, prompts, shared_ctx, tokenized_prompts, labels, Type="train"
    ):
        ### text branch
        text_embed = prompts + self.positional_embedding.type(self.dtype)
        text_embed = text_embed.permute(1, 0, 2)

        #  text_deeper_prompt = nn.ParameterList([nn.Parameter(torch.empty(self.nctx, 512)) for _ in range(self.prompt_depth - 1)])
        #  for single_para in text_deeper_prompt:
        #      nn.init.normal_(single_para, std = 0.02)
        ###

        ### image branch
        x = self.visual.conv1(image)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [
                self.visual.class_embedding.to(x.dtype)
                + torch.zeros(
                    x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
                ),
                x,
            ],
            dim=1,
        )
        x += self.visual.positional_embedding.to(x.dtype)

        visual_ctx = shared_ctx.expand(x.shape[0], -1, -1)
        x = torch.cat([x, visual_ctx], dim=1)

        x = self.visual.ln_pre(x)
        img_embed = x.permute(1, 0, 2)

        vision_deeper_prompt = nn.ParameterList(
            [
                nn.Parameter(torch.empty(self.nctx, 768))
                for _ in range(self.prompt_depth - 1)
            ]
        )
        for single_para in vision_deeper_prompt:
            nn.init.normal_(single_para, std=0.02)
        ###

        for idx, _ in enumerate(self.transformer.resblocks):
            vision_mask = torch.zeros(size=(img_embed.size(1), img_embed.size(0)))
            if (
                Type == "train"
                and idx < self.mask_depth
                and self.mask_mode == "random_location"
            ):

                text_cls_token = (
                    text_embed[
                        tokenized_prompts.argmax(dim=-1),
                        torch.arange(text_embed.size(1)),
                    ]
                    @ self.text_projection_image
                )

                for i in range(len(labels)):

                    sim = text_cls_token @ img_embed[:197, i].T

                    if Type == "train":  # 训练时使用正式标签，测试时使用推理标签
                        row_index = labels[i]
                    else:
                        row_index = torch.argmin(sim[:, 0])

                    selected_row = sim[row_index, 1:]

                    row_mask = torch.zeros(size=(196,))
                    indices = torch.topk(
                        selected_row, int(196 * self.mask_ratio), largest=False
                    ).indices
                    row_mask[indices] = 1

                    random_row_mask = torch.zeros(size=(196,))
                    random_row_mask[: int(196 * self.mask_ratio)] = 1
                    j = torch.randperm(196)

                    # vision_mask[i, 1: 197] = row_mask.to(torch.int) & random_row_mask[j].to(torch.int)
                    vision_mask[i, 1:197] = row_mask.to(torch.int) | random_row_mask[
                        j
                    ].to(torch.int)
                    # vision_mask[i, 1: 197] = row_mask.to(torch.int)

            text_embed = self.transformer.resblocks[idx](text_embed)

            img_embed = self.visual.transformer.resblocks[idx](
                x=img_embed, key_padding_mask=vision_mask
            )

            if idx < self.prompt_depth - 1:
                text_prefix = text_embed[:1]
                text_suffix = text_embed[1 + self.nctx :]
                textual_context = text_embed[1 : 1 + self.nctx]

                vision_prefix = img_embed[: -1 * self.nctx]
                vision_context = img_embed[-1 * self.nctx :]

                aug_textual_context = self.image_projections_prompt[idx](
                    textual_context, vision_context
                )
                # aug_textual_context = text_deeper_prompt[idx].to(self.config['device'])
                # aug_textual_context = aug_textual_context.unsqueeze(1).expand(-1, text_prefix.shape[1], -1)

                # aug_vision_context = self.prompt_projections_image[idx](vision_context, textual_context)
                aug_vision_context = vision_deeper_prompt[idx].to(img_embed.device)
                aug_vision_context = aug_vision_context.unsqueeze(1).expand(
                    -1, vision_prefix.shape[1], -1
                )

                text_embed = torch.cat(
                    [text_prefix, aug_textual_context, text_suffix], dim=0
                )
                img_embed = torch.cat([vision_prefix, aug_vision_context], dim=0)

        ### image branch
        x = img_embed.permute(1, 0, 2)
        x = self.visual.ln_post(x[:, 0])

        img_proj = x @ self.visual.proj
        ###

        ### text branch
        text_embed = text_embed.permute(1, 0, 2)
        text_embed = self.ln_final(text_embed)
        text_embed = (
            text_embed[
                torch.arange(text_embed.shape[0]), tokenized_prompts.argmax(dim=-1)
            ]
            @ self.text_projection
        )
        ###

        return img_proj, text_embed


class ProjHead(nn.Module):
    def __init__(self, in_dim=512, mlp_dim=4096, out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, mlp_dim),
            nn.BatchNorm1d(mlp_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_dim, out_dim),
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

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class MultiPromptLearner(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()

        classnames = cfg["dataset"]["classname"]
        self.n_cls = len(classnames)
        self.n_ctx = cfg["model"]["nctx"]
        ctx_init = cfg["model"]["language_init"]

        ctx_dim = 512
        vis_dim = 768

        if ctx_init and (self.n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt)
            ctx_vectors = embedding[0, 1 : 1 + self.n_ctx, :]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(self.n_ctx, ctx_dim)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * self.n_ctx)

        self.ctx = nn.Parameter(ctx_vectors)

        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = clip.tokenize(prompts)  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer(
            "token_suffix", embedding[:, 1 + self.n_ctx :, :]
        )  # CLS, EOS

        self.tokenized_prompts = tokenized_prompts  # torch.Tensor

        # vision proj
        self.proj = nn.Linear(ctx_dim, vis_dim)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (during training) or n_cls (during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim)
                ctx,  # (dim0, n_ctx, dim)
                suffix,  # (dim0, *, dim)
            ],
            dim=1,
        )

        return prompts

    def forward(self):
        prefix = self.token_prefix
        suffix = self.token_suffix
        ctx = self.ctx  # (n_ctx, ctx_dim)

        prompts = self.construct_prompts(ctx, prefix, suffix)
        shared_ctx = self.proj(self.ctx)

        return prompts, shared_ctx


class MEFAS(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        clip_mefas_config = {}

        if cfg["model"]["backbone"] == "ViT-B/16":
            model_path = r"/home/myw/.cache/clip/ViT-B-16.pt"
            clip_mefas_config = {
                "embed_dim": 512,
                "image_resolution": 224,
                "vision_layers": 12,
                "vision_width": 768,
                "vision_patch_size": 16,
                "context_length": 77,
                "vocab_size": 49408,
                "transformer_width": 512,
                "transformer_heads": 8,
                "transformer_layers": 12,
                "nctx": cfg["model"]["nctx"],  # len of soft prompt
                "prompt_depth": cfg["model"][
                    "prompt_depth"
                ],  # Max 12, Min 1, for 1 it will act as shallow MaPLe (J = 1)
                "mask_mode": cfg["model"]["mask"]["mode"],
                "mask_ratio": cfg["model"]["mask"]["ratio"],
                "mask_depth": cfg["model"]["mask"]["depth"],
            }

        self.clipModel = CLIP_MEFAS(clip_mefas_config)

        state_dict = torch.jit.load(model_path)

        self.clipModel.load_state_dict(state_dict.state_dict(), strict=False)

        self.prompt_learner = MultiPromptLearner(cfg, self.clipModel)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts

        self.logit_scale = self.clipModel.logit_scale
        self.projection_head = ProjHead(in_dim=512, mlp_dim=2048, out_dim=256)

        # print("Turning off gradients in both the image and the text encoder")
        # name_to_update = ['prompt_learner', 'classifier_fc', 'prompt_projections_image', 'image_projections_prompt', 'projection_head', 'text_projection_image']

        # for name, param in self.clip_model.named_parameters():
        #     if not any(name_part in name for name_part in name_to_update):
        #         param.requires_grad_(False)

    def forward(self, input_dict) -> Dict:
        img = input_dict["img"]
        aug1_img = input_dict["aug1_img"]
        aug2_img = input_dict["aug2_img"]
        labels = input_dict["label"]
        isTrain = input_dict["isTrain"]

        prompts, shard_ctx = self.prompt_learner()

        image = torch.cat([img, aug1_img, aug2_img], dim=0)

        img_feat, text_feat = self.clipModel(
            image=image,
            prompts=prompts,
            shared_ctx=shard_ctx,
            tokenized_prompts=self.tokenized_prompts,
            labels=labels,
            Type="train" if isTrain else "test",
        )

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        origin_feat = img_feat[: img.shape[0]]
        aug1_feat = img_feat[img.shape[0] : img.shape[0] + aug1_img.shape[0]]
        aug2_feat = img_feat[-1 * aug2_img.shape[0] :]

        ### SimCLR features
        out_aug1_feat = self.projection_head(aug1_feat)
        out_aug2_feat = self.projection_head(aug2_feat)
        ###

        ### cls loss
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * origin_feat @ text_feat.t()
        ###

        ### MSE loss
        text_embedding_v1 = []
        text_embedding_v2 = []

        for label in labels:

            if label == 0:
                text_embedding_v1.append(text_feat[0])
                text_embedding_v2.append(text_feat[0])

            elif label == 1:
                text_embedding_v1.append(text_feat[1])
                text_embedding_v2.append(text_feat[1])

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
