import torch
import torch.nn as nn
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

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=192):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        x = self.proj(x)  # Shape: (batch_size, embed_dim, H', W')
        x = x.flatten(2)  # Shape: (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2)  # Shape: (batch_size, num_patches, embed_dim)
        return x
    
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim=192, num_heads=3, mlp_ratio=4.0, dropout=0.1, drop_path=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first = True)
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
        x = x + self.drop_path1(self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0])
        x = x + self.drop_path2(self.mlp(self.ln2(x)))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_classes=1000,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=4.0, dropout=0.1, drop_path_rate=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.default_h, self.default_w = img_size // patch_size, img_size // patch_size
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Stochastic depth rate for each Transformer block
        drop_path_rates = [drop_path_rate * (i / (depth - 1)) for i in range(depth)]
        self.blocks = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_ratio, dropout, drop_path=drop_path_rates[i])
            for i in range(depth)
        ])
        
        self.ln = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        self.initialize_weights()

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
                
    def interpolate_pos_encoding(self, h, w):
        """对位置编码进行插值以适应不同的输入大小"""
        pos_embed = self.pos_embed # [bath_size, num_patches, embed_dim]
        N = pos_embed.shape[0]
        
        if h == self.default_h and w == self.default_w:
            return pos_embed
        
        # 首先将一维位置编码重塑为2D
        pos_embed = pos_embed.reshape(1, self.default_h, self.default_w, -1).permute(0, 3, 1, 2)
        # 使用双线性插值调整大小
        pos_embed = F.interpolate(
            pos_embed, 
            size=(h, w), 
            mode='bilinear',
            align_corners=False
        )
        
        # 重新整形为序列
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, h * w, -1)
        return pos_embed
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        
        h, w = H // self.patch_size, W // self.patch_size
        pos_embed = self.interpolate_pos_encoding(h, w)
        
        x = x + pos_embed
        x = self.pos_drop(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.ln(x)
        cls_token = torch.mean(x, dim = 1)
        
        return self.head(cls_token)