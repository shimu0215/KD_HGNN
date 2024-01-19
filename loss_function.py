import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitLoss(nn.Module):
    def __init__(self):
        super(LogitLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.kld_loss = nn.KLDivLoss()

    def forward(self, prediction_train, prediction, soft_target, label, alpha=0.5):
        loss1 = self.cross_entropy_loss(prediction_train, label)

        log_probs = F.log_softmax(prediction, dim=1)
        loss2 = self.kld_loss(log_probs, F.softmax(soft_target, dim=1))

        # total_loss = alpha * loss1 + (1 - alpha) * loss2
        return loss1, loss2

class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, prediction, target, student_sim_ind):

        loss = 0

        for i in range(len(prediction)):

            # loss += self.mse_loss(prediction[0], target[0][student_sim_ind[0]])
            loss += self.mse_loss(prediction[i], target[i][student_sim_ind[i]])

        return loss / len(prediction)

class EmbeddingLoss(nn.Module):
    def __init__(self):
        super(EmbeddingLoss, self).__init__()
        # self.mse_loss = nn.CosineEmbeddingLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, prediction, target):

        # loss = self.mse_loss(prediction[-1], target, torch.ones(target.size()[0]))
        loss = self.mse_loss(prediction[-1], target)

        return loss
