import torch
from models.base_model import BasicClassifier
import torch.nn as nn

class LogitCalibratedLoss(nn.Module):
    def __init__(self, tau, clients_label_counts, client_id):
        super().__init__()
        self.tau = tau
        self.clients_label_counts = clients_label_counts
        self.client_id = client_id

    def forward(self, logits, targets):
        cal_logit = torch.exp(
            logits - (
                self.tau
                * torch.pow(self.clients_label_counts[self.client_id], -1 / 4)
                .unsqueeze(0)
                .expand(logits.shape[0], -1)
            )
        )
        y_logit = torch.gather(cal_logit, dim=-1, index=targets.unsqueeze(1))
        loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
        return loss.mean()


class CNNForTesting(BasicClassifier):
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        spatial_dims: int = 3,
        loss=nn.BCEWithLogitsLoss,
        loss_kwargs: dict = {},
        optimizer=torch.optim.AdamW,
        optimizer_kwargs: dict = {'lr': 1e-4},
        lr_scheduler=None,
        lr_scheduler_kwargs: dict = {},
        aucroc_kwargs: dict = {"task": "binary"},
        acc_kwargs: dict = {"task": "binary"}
    ):
        super().__init__(in_ch, out_ch, spatial_dims, loss, loss_kwargs, optimizer, optimizer_kwargs,
                         lr_scheduler, lr_scheduler_kwargs, aucroc_kwargs, acc_kwargs)

    def forward(self, x_in: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x_in)


class MiniCNNForTesting(CNNForTesting):
    def __init__(self, loss=None, **kwargs):
        super().__init__(**kwargs)
        self.loss = loss or nn.BCEWithLogitsLoss()
        self.model = nn.Sequential(
            nn.Conv2d(1, 3, 3),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 4 * 4, 1)
        )

    def forward(self, x):
        return self.model(x)


class FixedSizeCNNForTesting(CNNForTesting):
    def __init__(self, artificial_model_size: int):
        super().__init__()
        float_size = 2  # oder 4, je nach GPU
        heuristic_factor = 1.03
        linear_size = int((artificial_model_size / float_size) ** 0.5 / heuristic_factor)

        self.model = nn.Sequential(
            nn.Conv2d(1, 3, 3),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 4 * 4, linear_size),
            nn.Linear(linear_size, linear_size),
            nn.Linear(linear_size, 1)
        )