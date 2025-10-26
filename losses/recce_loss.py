import torch
from torch import nn
import torch.nn.functional as F


class RecceReconsLoss(nn.Module):
    def __init__(self, reduction=None):
        super().__init__()
        self.reduction = reduction

    def forward(self, recons_x, x, y):
        recons_x = recons_x[0]
        loss = torch.tensor(0.0, device=x.device)

        real_index = torch.where(1 - y)[0]  # 0 - real, 1 - fake, 获取real samples index

        for index in real_index:

            real_x = torch.index_select(x, dim=0, index=real_index)
            real_rec = torch.index_select(recons_x, dim=0, index=real_index)
            real_rec = F.interpolate(
                real_rec, size=x.shape[-2:], mode="bilinear", align_corners=True
            )

            if self.reduction == "mean":
                loss += torch.mean(torch.abs(real_rec - real_x))

        return loss


class RecceMLLoss(nn.Module):  # metric-learning loss
    def __init__(self, reduction=None):
        super().__init__()

    def forward(self, input, target, eps=1e-6):
        # 0 - real; 1 - fake.
        """
        区分(real, fake) samples, 通过优化real与fake差异性
        args:
        input (Tensor): recce中decoder多层feature

        """
        loss = torch.tensor(0.0, device=target.device)
        batch_size = target.shape[0]
        mat_1 = torch.hstack([target.unsqueeze(-1)] * batch_size)
        mat_2 = torch.vstack([target] * batch_size)
        diff_mat = torch.logical_xor(mat_1, mat_2).float()
        or_mat = torch.logical_or(mat_1, mat_2)
        eye = torch.eye(batch_size, device=target.device)
        or_mat = torch.logical_or(or_mat, eye).float()
        sim_mat = 1.0 - or_mat
        for _ in input:
            diff = torch.sum(_ * diff_mat, dim=[0, 1]) / (
                torch.sum(diff_mat, dim=[0, 1]) + eps
            )
            sim = torch.sum(_ * sim_mat, dim=[0, 1]) / (
                torch.sum(sim_mat, dim=[0, 1]) + eps
            )
            partial_loss = 1.0 - sim + diff
            loss += max(partial_loss, torch.zeros_like(partial_loss))
        return loss
