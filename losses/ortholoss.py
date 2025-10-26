import torch
from torch import nn
import torch.nn.functional as F

class OrthoLoss(nn.Module):
    def __init__(self, reduction = None):
        super().__init__()

    def forward(self, low, mid, high):

        bs = low.shape[0]

        low = F.normalize(low, p = 2, dim = 2)
        mid = F.normalize(mid, p = 2, dim = 2)
        high = F.normalize(high, p = 2, dim = 2)

        low_mid_gram = torch.bmm(low, mid.transpose(1, 2))
        low_high_gram = torch.bmm(low, high.transpose(1, 2))
        mid_high_gram = torch.bmm(mid, high.transpose(1, 2))

        total_loss = (low_mid_gram ** 2).sum() + (low_high_gram ** 2).sum() + (mid_high_gram ** 2).sum()

        return total_loss / (bs * 3)

