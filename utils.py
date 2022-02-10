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
     noise = torch.clip(torch.randn_like(inputs)*0.01, min=0, max=1)
     return inputs + noise


def accuracy(y_pred, y_true):
    y_pred = torch.round(y_pred)
    y_true = torch.round(y_true)
    right = (y_pred == y_true)
    return (torch.sum(right) / len(right))


def minmax_scale(v, new_min, new_max):
    with torch.no_grad():
        v_min, v_max = v.min(), v.max()
        v = (v - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
    return v
    