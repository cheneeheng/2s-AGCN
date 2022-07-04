import torch
import torch.nn.functional as F

from typing import Tuple


# From: https://github.com/microsoft/SGN/blob/master/main.py
class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, classes: int, smoothing=0.0, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


# From: https://github.com/stnoah1/infogcn/blob/master/loss.py
class MaximumMeanDiscrepancyLoss(torch.nn.Module):
    def __init__(self, classes: int):
        super().__init__()
        self.cls = classes

    def forward(self, z: torch.Tensor, z_prior: torch.Tensor, y: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        y_valid = [i_cls in y for i_cls in range(self.cls)]
        z_mean = torch.stack(
            [z[y == i_cls].mean(dim=0)  # mean over the batch with that GT cls
             for i_cls in range(self.cls)],
            dim=0
        )
        # see section 3.4
        l2_z_mean = torch.linalg.norm(z.mean(dim=0), ord=2)
        mmd_loss = F.mse_loss(z_mean[y_valid], z_prior[y_valid].to(z.device))
        return mmd_loss, l2_z_mean, z_mean[y_valid]
