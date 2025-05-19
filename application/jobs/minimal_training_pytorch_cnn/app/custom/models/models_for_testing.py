from models.base_model import BasicClassifier
import torch
import torch.nn as nn

class MiniCNNForTesting(BasicClassifier):
    def __init__(self):
     
        def logit_calibrated_loss(logits, targets):
            tau = 1.0  
            num_classes = logits.shape[1]
            device = logits.device
        
            label_counts = torch.ones(num_classes, device=device)
            cal_logits = torch.exp(
                logits - (tau * torch.pow(label_counts, -1 / 4).unsqueeze(0).expand(logits.shape))
            )
            y_logit = torch.gather(cal_logits, dim=-1, index=targets.unsqueeze(1))
            loss = -torch.log(y_logit / cal_logits.sum(dim=-1, keepdim=True))
            return loss.mean()

        super().__init__(
            in_ch=1,
            out_ch=2,           
            spatial_dims=3,
            loss=logit_calibrated_loss
        )

        self.model = torch.nn.Sequential(
            nn.Conv2d(1, 3, 3),
            nn.ReLU(),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3*4*4, 2)    
        )