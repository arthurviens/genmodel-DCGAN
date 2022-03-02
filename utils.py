import torch
import os
import pandas as pd

def init_weights(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)

def init_ortho(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.orthogonal_(m.weight)

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
    return (torch.sum(right) / len(right)).item()


def minmax_scale(v, new_min, new_max):
    with torch.no_grad():
        v_min, v_max = v.min(), v.max()
        v = (v - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
    return v


def apply_weight_decay(*modules, weight_decay_factor=0., wo_bn=True):
    '''
    https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/5
    Apply weight decay to pytorch model without BN;
    In pytorch:
        if group['weight_decay'] != 0:
            grad = grad.add(p, alpha=group['weight_decay'])
    p is the param;
    :param modules:
    :param weight_decay_factor:
    :return:
    '''
    for module in modules:
        for m in module.modules():
            if hasattr(m, 'weight'):
                if wo_bn and isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    continue
                m.weight.grad += m.weight * weight_decay_factor

    

def write_params(p, folder='saved_models', verbose=0):
    filename = p['filename']
    string = f"Name : {p['filename']}\n########### GLOBAL ###########\nDS: {p['ds']}\nRun test : {p['run_test']}\nBatch size : {p['bs']}\nCrop_size : {p['crop_size']}\n\n"
    string += f"########### ARCHI ###########\nInput dim : {p['z_dim']}\n{p['archi_info']}\n\n"
    string += "########### TRAINING PARAMS ###########\n"
    string += f"Epochs : {p['n_epoch']}\nSave freq : {p['save_frequency']}\nDiscriminator learning factor (k) : {p['k']}\n\n"
    string += f"########### MODEL PARAMS ###########\nlrG : {p['lrG']}\nlrD : {p['lrD']}\nbeta : {p['beta1']}\nWeight decay Discriminator : {p['weight_decayD']}\nWeight decay Generator : {p['weight_decayG']}\n"
    string += f"label_reals : {p['label_reals']}\nlabel_fakes :{p['label_fakes']}\n\n"
    string += f"########### LAST EPOCH ###########\n"
    string += f"Last epoch : {p['epoch']}"

    if verbose:
        print(string)
        print()
        print("#######################\n")

    filename += "-PARAMS"
    with open(os.path.join(folder, filename), 'w+') as file :
        file.write(string)
        file.close()


def get_epoch_from_log(param_dict, folder='saved_models', verbose=1):
    with open(os.path.join(folder, param_dict["filename"] + "-PARAMS"), "r") as f:
        lines = pd.Series(f.readlines())
    #### Verif paramètres égaux TODO ###

    try:
        epoch_line = lines[lines.str.startswith("Last epoch")]
        epoch = int(epoch_line.values[0].split(":")[1])
        param_dict['epoch'] = epoch
    except Exception as e:
        print(f"Could not retrieve epoch from log : {e}")