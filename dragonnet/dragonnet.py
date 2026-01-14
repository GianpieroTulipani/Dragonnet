import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = torch.sum((1. - t_true) * torch.square(y_true - y0_pred))
    loss1 = torch.sum(t_true * torch.square(y_true - y1_pred))

    return loss0 + loss1

def binary_classification_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]

    t_pred = torch.clamp(t_pred, 1e-6, 1 - 1e-6)

    losst = F.binary_cross_entropy(
        t_pred,
        t_true,
        reduction='sum'
    )

    return losst

def dragonnet_loss(concat_true, concat_pred):
    return regression_loss(concat_true, concat_pred) + binary_classification_loss(concat_true, concat_pred)


def tarreg_loss(ratio, concat_true, concat_pred):
    vanilla_loss = dragonnet_loss(concat_true, concat_pred)

    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]
    t_pred = concat_pred[:, 2]

    eps = concat_pred[:, 3]
    t_pred = torch.clamp(t_pred, 1e-6, 1 - 1e-6)

    y_pred = t_true * y1_pred + (1 - t_true) * y0_pred

    h = t_true / t_pred - (1 - t_true) / (1 - t_pred)

    y_pert = y_pred + eps * h

    targeted_regularization = torch.sum((y_true - y_pert) ** 2)

    loss = vanilla_loss + ratio * targeted_regularization
    return loss

class DatasetACIC(Dataset):
    def __init__(self, x, t, y):
        super().__init__()
        self.X = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.t = torch.tensor(t, dtype=torch.float32)
       
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]
        t = self.t[index]
        return x, y, t

class EpsilonLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.epsilon = nn.Parameter(torch.randn(1, 1), requires_grad=True)

    def forward(self, t_pred):
        return self.epsilon.expand_as(t_pred)

class Dragonnet(nn.Module):
    def __init__(self, input_dim, rep_units=256, head_units=128, dropout_p=0.2):
        """
        input_dim: int, number of input covariates
        rep_units: int, number of units in shared representation layers
        head_units: int, number of units in outcome heads
        dropout_p: float, dropout probability
        """
        super().__init__()
        self.dropout_p = dropout_p

        self.rep = nn.Sequential(
            nn.Linear(input_dim, rep_units),
            nn.ELU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(rep_units, rep_units),
            nn.ELU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(rep_units, rep_units),
            nn.ELU(),
            nn.Dropout(p=self.dropout_p)
        )

        self.t_head = nn.Sequential(
            nn.Linear(rep_units, 1),
            nn.Sigmoid()
        )

        self.y0_head = nn.Sequential(
            nn.Linear(rep_units, head_units),
            nn.ELU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(head_units, head_units),
            nn.ELU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(head_units, 1)
        )

        self.y1_head = nn.Sequential(
            nn.Linear(rep_units, head_units),
            nn.ELU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(head_units, head_units),
            nn.ELU(),
            nn.Dropout(p=self.dropout_p),
            nn.Linear(head_units, 1)
        )

        self.epsilon_layer = EpsilonLayer()

    def forward(self, x):
        x = self.rep(x)

        t_pred = self.t_head(x)
        y0_pred = self.y0_head(x)
        y1_pred = self.y1_head(x)

        eps = self.epsilon_layer(t_pred)

        concat_pred = torch.cat([y0_pred, y1_pred, t_pred, eps], dim=1)
        return concat_pred