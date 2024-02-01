import torch
import torch.nn as nn
import torch.nn.functional as F


class LogitLoss(nn.Module):
    def __init__(self):
        super(LogitLoss, self).__init__()
        self.kld_loss = nn.KLDivLoss()

    def forward(self, prediction, soft_target):
        t = 1
        log_probs = F.log_softmax(prediction/t, dim=1)
        loss = self.kld_loss(log_probs, F.softmax(soft_target/t, dim=1))

        return loss


class GtLoss(nn.Module):
    def __init__(self):
        super(GtLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, prediction_train, label):
        loss = self.cross_entropy_loss(prediction_train, label)

        return loss


class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()
        # self.mse_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

    def forward(self, prediction, target, student_sim_ind):
        predictions = prediction[0]
        targets = target[0][student_sim_ind[0]]

        for i in range(1, len(prediction)):
            predictions = torch.cat([predictions, prediction[i]])
            targets = torch.cat([targets, target[i][student_sim_ind[i]]])

        struc_loss = self.mse_loss(predictions, targets)

        return struc_loss


class EmbeddingLoss(nn.Module):
    def __init__(self):
        super(EmbeddingLoss, self).__init__()
        self.cos_loss = nn.CosineEmbeddingLoss()
        # self.mse_loss = nn.MSELoss()

    def forward(self, prediction, target):
        loss = self.cos_loss(prediction[-1], target, torch.ones(target.size()[0]))
        # loss = self.mse_loss(prediction[-1], target)

        return loss
