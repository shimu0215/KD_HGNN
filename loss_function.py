import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.kld_loss = nn.KLDivLoss()

    def forward(self, prediction_train, prediction, soft_target, label, alpha=0.5):
        loss1 = self.cross_entropy_loss(prediction_train, label)

        log_probs = F.log_softmax(prediction)
        loss2 = self.kld_loss(log_probs, F.softmax(soft_target))

        total_loss = alpha * loss1 + (1 - alpha) * loss2
        return total_loss
