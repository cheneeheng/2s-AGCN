import torch
import torch.nn.functional as F

from typing import Tuple, Optional


class CosineLoss(torch.nn.CosineSimilarity):
    def __init__(self, mode: int = 1):
        self.mode = mode
        if self.mode == 2:
            dim = -1
        else:
            dim = 1
        super().__init__(dim=dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        if self.mode == 2:
            x1 = torch.linalg.norm(x1, ord=2, dim=1)
            x2 = torch.linalg.norm(x2, ord=2, dim=1)
        loss = super().forward(x1, x2)
        return 1.0 - torch.mean(loss)


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


# Based on:
# https://github.com/umbertogriffo/focal-loss-keras/blob/master/src/loss_function/losses.py
# https://github.com/artemmavrin/focal-loss/blob/master/src/focal_loss/_categorical_focal_loss.py
class CategorialFocalLoss(torch.nn.Module):
    def __init__(self,
                 classes: int,
                 smoothing=0.0,
                 alpha: Optional[torch.Tensor] = None,
                 gamma: float = 2):
        super().__init__()
        self.eps = smoothing / classes
        self.confidence = 1.0 - smoothing + self.eps
        self.gamma = gamma
        if alpha is None:
            self.alpha = None
        elif len(alpha) == 0:
            self.alpha = None
        else:
            self.alpha = alpha
            assert alpha.shape[0] == classes

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # https://discuss.pytorch.org/t/what-is-the-formula-for-cross-entropy-loss-with-label-smoothing/149848
        # the formulation here is slightly different than the one used
        # in `LabelSmoothingLoss()` above.
        # CE
        log_probs = pred.log_softmax(dim=-1)  # N,C
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)  # N,C
            true_dist.fill_(self.eps)
            true_dist.scatter_(1,
                               target.data.unsqueeze(1),
                               self.confidence)  # N
        ce_loss = torch.sum(-true_dist * log_probs, dim=-1)  # N

        # alpha is a form of class weighting.
        if self.alpha is not None:
            ce_loss *= torch.gather(self.alpha, -1, target)

        # FL
        probs = F.softmax(pred, dim=-1)  # N,C
        probs = torch.gather(probs, -1, target.data.unsqueeze(1)).squeeze(1)
        focal_modulation = (1 - probs) ** self.gamma  # N
        focal_loss = focal_modulation * ce_loss  # N
        return torch.mean(focal_loss)


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


if __name__ == '__main__':
    weight = torch.Tensor([1, 2, 3, 4, 5, 6]).cuda()
    CFL = CategorialFocalLoss(6, 0.1, weight, 0.5)
    logits = torch.rand(4, 6).cuda()
    target = torch.Tensor([2, 4, 3, 1]).type(torch.int64).cuda()
    losses = CFL(logits, target)
    print(logits, target, losses)
