import torch

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def add_noise(inputs):
     noise = torch.clip(torch.randn_like(inputs), min=-1, max=1)
     return inputs + noise


def accuracy(y_pred, y_true):
    y_pred = torch.round(y_pred)
    right = (y_pred == y_true)
    return (torch.sum(right) / len(right))

