# band内token使用动态参数
import torch
import einops
import torch.nn as nn
from fractions import Fraction
import torch.nn.functional as F
    
class DropPath(nn.Module):
    """DropPath (Stochastic Depth) regularization"""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class FrequencyBandAttention(nn.Module):
    def __init__(self, embed_dim=192, num_heads=3, dropout=0.1, lowfreq_range = 1 / 16, middlefreq_range = 1 / 8):
        super().__init__()
        self.num_heads = num_heads
        self.lowfreq_range = Fraction(lowfreq_range)
        self.middlefreq_range = Fraction(middlefreq_range)
        self.embed_dim = embed_dim
        self.dropout = nn.Dropout(dropout)
        
        self.intra_band_attn_layer = nn.MultiheadAttention(embed_dim, num_heads, dropout = dropout, batch_first = True)
        
        self.inter_band_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout = dropout, batch_first = True)

    def split_bands(self, x):
        """
        Split the input tokens into low, middle, and high frequency bands based on ranges.
        """
        batch_size, token_num, _ = x.shape
        lowfreq_cutoff = int(token_num * self.lowfreq_range)
        middlefreq_cutoff = int(token_num * self.middlefreq_range)
        
        lowfreq_tokens = x[:, :lowfreq_cutoff, :]
        middlefreq_tokens = x[:, lowfreq_cutoff: middlefreq_cutoff, :]
        highfreq_tokens = x[:, middlefreq_cutoff: , :]
        
        return lowfreq_tokens, middlefreq_tokens, highfreq_tokens

    def intra_band_attn(self, tokens):
        """
        Apply self-attention within each band (intra-band attention) using multihead attention.
        """
        attn_output, _ = self.intra_band_attn_layer(tokens, tokens, tokens)
        return attn_output

    def forward(self, x):
        """
        x shape: [batch_size, token_num, embed_dim]
        """
        # Step 1: Split tokens into low, middle, and high frequency bands
        lowfreq_tokens, middlefreq_tokens, highfreq_tokens = self.split_bands(x)
        
        # Step 2: Apply intra-band attention using multihead attention
        lowfreq_attn = self.intra_band_attn(lowfreq_tokens)
        middlefreq_attn = self.intra_band_attn(middlefreq_tokens)
        highfreq_attn = self.intra_band_attn(highfreq_tokens)
        
        # Step 3: Calculate the mean token for each frequency band
        lowfreq_mean = lowfreq_attn.mean(dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
        middlefreq_mean = middlefreq_attn.mean(dim=1, keepdim=True)  # [batch_size, 1, embed_dim]
        highfreq_mean = highfreq_attn.mean(dim=1, keepdim=True)  # [batch_size, 1, embed_dim]

                
        # Combine band means for inter-band attention
        band_means = torch.cat([lowfreq_mean, middlefreq_mean, highfreq_mean], dim=1)  # [batch_size, 3, embed_dim]
        
        # Step 4: Apply inter-band attention using multihead attention
        band_attn_output, _ = self.inter_band_attn(band_means, band_means, band_means)
        
        # Step 5: Assign band attention output back to original tokens, with residual connection
        lowfreq_combined = lowfreq_attn - lowfreq_mean + band_attn_output[:, 0: 1, :]
        middlefreq_combined = middlefreq_attn - middlefreq_mean + band_attn_output[:, 1: 2, :]
        highfreq_combined = highfreq_attn - highfreq_mean + band_attn_output[:, 2: 3, :]
        
        # Step 6: Concatenate the frequency bands back together
        combined_output = torch.cat([lowfreq_combined, middlefreq_combined, highfreq_combined], dim = 1)
        
        return combined_output

    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4.0, dropout=0.1, drop_path=0.0, lowfreq_range = 1/16, middlefreq_range = 1/8):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = FrequencyBandAttention(embed_dim, num_heads, lowfreq_range = lowfreq_range, middlefreq_range = middlefreq_range, dropout = dropout)
        self.drop_path1 = DropPath(drop_path)
        
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout)
        )
        self.drop_path2 = DropPath(drop_path)
        
    def forward(self, x):
        x = x + self.drop_path1(self.attn(self.ln1(x)))
        x = x + self.drop_path2(self.mlp(self.ln2(x)))
        return x


class AnnularPatchEmbed(nn.Module):
    def __init__(self, img_size = 224, num_rings = 16, max_pool = True):
        super().__init__()
        self.img_size = img_size
        self.num_rings = num_rings
        self.max_pool = max_pool
        
        self.tokens_weights = nn.Parameter(torch.randn(256, 1, 224, 224) * torch.sqrt(torch.tensor(2.0 / (1 + 192))), requires_grad = True)

        self.id_fc = nn.Linear(in_features = 256, out_features = 192)  # 将同维度不同分布的token转为同维度同分布token

        self.masks = self.GetMasks(self.img_size, self.num_rings)

    def GetMasks(self, img_size, num_rings):
        h, w = img_size, img_size
        center = (w // 2, h // 2)
        radius = min(h, w) // 2
        radius_step = radius // self.num_rings

        masks = torch.zeros((num_rings, 224, 224))
        
        for i in range(num_rings):
            inner_radius = i * radius_step
            outer_radius = (i + 1) * radius_step
            
            masks[i] = self.annular_mask((h, w), center, inner_radius, outer_radius)
        return masks

   
    def annular_mask(self, shape, center, inner_radius, outer_radius):
        '''
        生成环形mask (torch实现).
        '''
        Y, X = torch.meshgrid(torch.arange(shape[0]), torch.arange(shape[1]), indexing='ij')
        dist_from_center = torch.sqrt((X - center[0]).float()**2 + (Y - center[1]).float()**2)
        mask = (dist_from_center >= inner_radius) & (dist_from_center < outer_radius)
        return mask
    
    def img2annular_tokens(self, batch_img):
        '''
        图像切分为扇形，并生成tokens.
        batch_img: 输入图像张量，形状为 [B, H, W]
        '''
        b, h, w = batch_img.shape

        imgs = einops.repeat(batch_img, 'b h w -> b repeat h w', repeat = self.num_rings)
        masks = einops.repeat(self.masks, 'num h w -> repeat num h w', repeat = b).to(batch_img.device)

        masks_imgs = masks * imgs

        masks_imgs = einops.rearrange(masks_imgs, 'b c h w -> (b c) 1 h w')
        tokens = F.conv2d(masks_imgs, self.tokens_weights)
        tokens = einops.rearrange(tokens, '(b c) d 1 1 -> b c d', b = b, c = self.num_rings)
        tokens = self.id_fc(tokens)
        return tokens

    def forward(self, x):  # x 的形状: [B, H, W]
        # 针对每个批次的样本进行扇形切分
        batch_tokens = self.img2annular_tokens(x)  # [B, num_sectors, min_length]

        return batch_tokens
    
class FreqFormer(nn.Module):
    def __init__(self, img_size=224, num_rings = 16, max_pool = True,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, dropout=0.1, drop_path_rate=0.0,
                 lowfreq_range = 1/ 4, middlefreq_range = 1 / 2):
        super().__init__()
        self.patch_embed = AnnularPatchEmbed(img_size = img_size, num_rings = num_rings, max_pool = max_pool)
 
        self.pos_embed = nn.Parameter(torch.zeros(1, num_rings, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.num_rings = num_rings
        self.lowfreq_range = Fraction(lowfreq_range)
        self.middlefreq_range = Fraction(middlefreq_range)
        self.band_weights = nn.Parameter(torch.ones((3, )), requires_grad = True)

        self.cutoff_point = self.Get_band_cutoff_point()
        
        # Stochastic depth rate for each Transformer block
        drop_path_rates = [drop_path_rate * (i / (depth - 1)) for i in range(depth)]
        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_ratio, dropout, drop_path=drop_path_rates[i], lowfreq_range = lowfreq_range, middlefreq_range = middlefreq_range)
            for i in range(depth)
        ])
        
        self.ln = nn.LayerNorm(embed_dim)
        
        self.initialize_weights()

    def Get_band_cutoff_point(self):
        lowfreq_cutoff = int(self.num_rings * self.lowfreq_range)
        midfreq_cutoff = int(self.num_rings * self.middlefreq_range)
        
        return [lowfreq_cutoff, midfreq_cutoff]

    def initialize_weights(self):
        # Initialize the weights for the layers
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Initialize patch embedding layer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
                
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln(x)

        lowfreq_tokens = x[:, : self.cutoff_point[0]]
        midfreq_tokens = x[:, self.cutoff_point[0]: self.cutoff_point[1]]
        highfreq_tokens = x[:, self.cutoff_point[1]: ]

        return lowfreq_tokens, midfreq_tokens, highfreq_tokens