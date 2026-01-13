import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
    
def regression_loss(concat_true, concat_pred):
    y_true = concat_true[:, 0]
    t_true = concat_true[:, 1]

    y0_pred = concat_pred[:, 0]
    y1_pred = concat_pred[:, 1]

    loss0 = torch.sum((1. - t_true) * (y_true - y0_pred) ** 2)
    loss1 = torch.sum(t_true * (y_true - y1_pred) ** 2)

    return loss0 + loss1

def binary_classification_loss(concat_true, concat_pred):
    t_true = concat_true[:, 1]
    t_pred = concat_pred[:, 2]

    t_pred = torch.clamp(t_pred, 1e-7, 1-1e-7)

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

    t_pred = torch.clamp(t_pred, 1e-7, 1-1e-7)

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
        self.epsilon = nn.Parameter(torch.randn(1, 1))

    def forward(self, t_pred):
        return self.epsilon.expand_as(t_pred)
    
    
class Dragonnet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.rep = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU(),
            nn.Linear(200, 200),
            nn.ELU()
        )

        self.t_head = nn.Sequential(
            nn.Linear(200, 1),
            nn.Sigmoid()
        )

        self.y0_head = nn.Sequential(
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 1)
        )

        self.y1_head = nn.Sequential(
            nn.Linear(200, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 1)
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