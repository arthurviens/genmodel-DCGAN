import torch
import os

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
    return (torch.sum(right) / len(right)).item()


def minmax_scale(v, new_min, new_max):
    with torch.no_grad():
        v_min, v_max = v.min(), v.max()
        v = (v - v_min)/(v_max - v_min)*(new_max - new_min) + new_min
    return v
    

def write_params(filename, archi_info, lrG, lrD, beta1,
                weight_decayD, weight_decayG, z_dim,
                n_epoch, save_frequency, k, label_fakes, label_reals,
                ds, run_test, bs, crop_size, folder='saved_models'):
                

    string = f"Name : {filename}\n########### GLOBAL ###########\nDS: {ds}\nRun test : {run_test}\nBatch size : {bs}\nCrop_size : {crop_size}\n\n"
    string += f"########### ARCHI ###########\nInput dim : {z_dim}\n{archi_info}\n\n"
    string += "########### TRAINING PARAMS ###########\n"
    string += f"Epochs : {n_epoch}\nSave freq : {save_frequency}\nDiscriminator learning factor (k) : {k}\n\n"
    string += f"########### MODEL PARAMS ###########\nlrG : {lrG}\nlrD : {lrD}\nbeta : {beta1}\nWeight decay (regularization) D/G : {weight_decayD} / {weight_decayG}\n"
    string += f"label_reals : {label_reals}\nlabel_fakes : {label_fakes}"
    
    print(string)

    filename += "-PARAMS"
    with open(os.path.join(folder, filename), 'w+') as file :
        file.write(string)
        file.close()
