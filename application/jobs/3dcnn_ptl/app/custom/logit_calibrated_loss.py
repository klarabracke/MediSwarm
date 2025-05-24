import torch
import torch.nn as nn

class LogitCalibratedLoss(nn.Module):
    def __init__(self, tau, clients_label_counts, client_id):
        super().__init__()
        self.tau = tau
        self.clients_label_counts = clients_label_counts
        self.client_id = client_id

    def forward(self, logits, targets):
        device = logits.device
        label_counts = self.clients_label_counts[self.client_id].to(device)
        targets = targets.to(device).long().view(-1)  # Wichtig: 1D LongTensor

        cal_logit = torch.exp(
            logits - (
                self.tau
                * torch.pow(label_counts, -1 / 4)
                .unsqueeze(0)
                .expand(logits.shape[0], -1)
            )
        )
        y_logit = torch.gather(cal_logit, dim=-1, index=targets.unsqueeze(1))
        loss = -torch.log(y_logit / cal_logit.sum(dim=-1, keepdim=True))
        return loss.mean()