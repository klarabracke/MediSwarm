from models.base_model import BasicClassifier
import torch
import torch.nn as nn
import math

# ----------- Deine robuste calibrating Lossfunktion -----------
def logit_calibrated_loss(logits, targets, tau=1.0, label_counts=None):
    if label_counts is None:
        label_counts = torch.ones(logits.shape[-1], device=logits.device)
    if logits.dim() == 1:
        logits = logits.unsqueeze(1)
    targets = targets.long().view(-1)
    num_classes = logits.shape[-1]
    # Defensive Index-Check für CUDA
    if targets.numel() == 0 or targets.max() >= num_classes or targets.min() < 0:
        print("[WARNING] Loss skipped: Target out of bounds for gather! Skipping Loss.")
        return torch.tensor(0., device=logits.device, requires_grad=True)
    cal_logit = torch.exp(
        logits - (tau * torch.pow(label_counts, -1/4).unsqueeze(0).expand_as(logits))
    )
    y_logit = torch.gather(cal_logit, dim=-1, index=targets.unsqueeze(1))
    loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
    return loss.mean()

# ----------- Modelle ----------------------------------
class CNNForTesting(BasicClassifier):
    def __init__(self,
                 in_ch: int = 1,
                 out_ch: int = 1,
                 spatial_dims: int = 3,
                 loss=logit_calibrated_loss,
                 loss_kwargs: dict = {},
                 optimizer=torch.optim.AdamW,
                 optimizer_kwargs: dict = {'lr': 1e-4},
                 lr_scheduler=None,
                 lr_scheduler_kwargs: dict = {},
                 aucroc_kwargs: dict = {"task": "binary"},
                 acc_kwargs: dict = {"task": "binary"},
                 **kwargs
                 ):
        super().__init__(
            in_ch=in_ch,
            out_ch=out_ch,
            spatial_dims=spatial_dims,
            loss=loss,
            loss_kwargs=loss_kwargs,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            aucroc_kwargs=aucroc_kwargs,
            acc_kwargs=acc_kwargs,
            **kwargs
        )

    def forward(self, x_in: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x_in)

class MiniCNNForTesting(CNNForTesting):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = torch.nn.Sequential(
            nn.Conv2d(1, 3, 3),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3*4*4, 1)
        )

class FixedSizeCNNForTesting(CNNForTesting):
    def __init__(self, artificial_model_size: int, **kwargs):
        super().__init__(**kwargs)
        float_size = 2
        heuristic_factor = 1.03
        linear_size = int(math.sqrt(artificial_model_size/float_size)/heuristic_factor)

        self.model = torch.nn.Sequential(
            nn.Conv2d(1, 3, 3),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3*4*4, linear_size),
            nn.Linear(linear_size, linear_size),
            nn.Linear(linear_size, 1)
        )