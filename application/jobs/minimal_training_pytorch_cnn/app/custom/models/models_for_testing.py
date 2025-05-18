from models.base_model import BasicClassifier
import torch
import torch.nn as nn
import math

# ===== NEUE LOSS FUNKTION =====
def logit_calibrated_loss(logits, targets, tau=1.0, label_counts=None):
    """
    Einfache Form des Logit-Calibrated-Loss.
    Standardisiert für binary classification. 
    logits: output from model, shape [batch, 1] oder [batch, num_classes]
    targets: shape [batch, 1] oder [batch]
    tau: Skalierungsfaktor
    label_counts: Tensor mit Klassenhäufigkeiten (hier optional und als 1en gesetzt).
    """
    if label_counts is None:
        label_counts = torch.ones(logits.shape[-1], device=logits.device)
    if logits.dim() == 1:
        logits = logits.unsqueeze(1)
    

    targets = targets.long().view(-1)
    cal_logit = torch.exp(
        logits - (tau * torch.pow(label_counts, -1/4).unsqueeze(0).expand_as(logits))
    )
    y_logit = torch.gather(cal_logit, dim=-1, index=targets.unsqueeze(1))
    loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
    return loss.mean()

# ====== MODELLE ======
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
                 acc_kwargs: dict = {"task": "binary"}
                 ):
        super().__init__(in_ch, out_ch, spatial_dims, loss, loss_kwargs, optimizer, optimizer_kwargs, lr_scheduler,
                         lr_scheduler_kwargs, aucroc_kwargs, acc_kwargs)

    def forward(self, x_in: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(x_in)

class MiniCNNForTesting(CNNForTesting):
    def __init__(self):
        super().__init__()

        self.model = torch.nn.Sequential(
            nn.Conv2d(1, 3, 3),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3*4*4, 1)
        )

class FixedSizeCNNForTesting(CNNForTesting):
    def __init__(self,
                 artificial_model_size: int):
        super().__init__()

        float_size = 2   # 2 or 4, depending on float size on GPU
        heuristic_factor = 1.03  # to compensate for approximate formula
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