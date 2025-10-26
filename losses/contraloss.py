import torch
import einops
from torch import nn
import torch.nn.functional as F

class ContraLoss(nn.Module):
    def __init__(self, reduction = None, temperature = 0.1):
        super().__init__()

        self.temperature = temperature

    def forward(self, diff_freq_cls, diff_freq_text):

        logits = torch.matmul(diff_freq_cls, diff_freq_text.T) / self.temperature
        logits = einops.rearrange(logits, 'b n c -> (b n) c')
        labels = torch.arange(3).to(diff_freq_cls.device)
        labels = einops.repeat(labels, 'n -> (repeat n)', repeat = diff_freq_cls.shape[0])
        
        loss = nn.CrossEntropyLoss()(logits, labels)

        return loss



