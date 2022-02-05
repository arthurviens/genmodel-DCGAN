import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from dataload import *
from models import *
from tqdm import tqdm
import pandas as pd

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bs = 16
batch_size_test = 16

train_loader, test_loader = define_landscapes_loaders(bs, batch_size_test, 
                                                      rescale=256,
                                                      crop=224,
                                                      rgb=True)


lr = 0.00005
n_epoch = 100

# build network
z_dim = 128
landscape_dim = 224*224

G = Generator_224(z_dim).to(device)
D = Discriminator_224().to(device)

# print(G)
# print(D)

# loss
criterion = nn.BCELoss() 

# optimizer
G_optimizer = optim.Adam(G.parameters(), lr = lr)
D_optimizer = optim.Adam(D.parameters(), lr = lr)



def D_train(x):
    #=======================Train the discriminator=======================#
    D.zero_grad()


    # train discriminator on real
    x_real = x.view(-1, 3, 224 , 224).to(device)
    size = len(x_real)

    y_real = torch.ones(size, 1).to(device)

    D_output = D(x_real)
    D_real_loss = criterion(D_output, y_real)
    D_real_score = D_output

    # train discriminator on facke
    z = torch.randn(size, z_dim).to(device)
    x_fake, y_fake = G(z), torch.zeros(size, 1).to(device)

    D_output = D(x_fake)
    D_fake_loss = criterion(D_output, y_fake)
    D_fake_score = D_output

    # gradient backprop & optimize ONLY D's parameters
    D_loss = D_real_loss + D_fake_loss
    D_loss.backward()
    D_optimizer.step()

    return  D_loss.data.item()

def G_train(x):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(bs, z_dim).to(device)
    y = torch.ones(bs, 1).to(device)

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, y)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
    

    return G_loss.data.item()


if __name__ == "__main__":
    D_losses, G_losses = [], []
    savefile = 'gan'

    for epoch in range(1, n_epoch+1):           
        D_current_loss, G_current_loss = 0.0, 0.0
        count = 0

        for batch_idx, (x) in enumerate(tqdm(train_loader)):
            D_current_loss += D_train(x)
            G_current_loss += G_train(x)
            count += len(x)

        D_current_loss /= count
        G_current_loss /= count

        D_losses.append(D_current_loss)
        G_losses.append(G_current_loss)

        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
                (epoch), n_epoch, D_current_loss, G_current_loss))

        if (savefile is not None) and (epoch % 10 == 0) and (epoch > 0):
            torch.save(G.state_dict(), f"saved_models/{savefile}_generator.sav")
            torch.save(D.state_dict(), f"saved_models/{savefile}_discriminator.sav")
            pd.DataFrame(data=np.array([D_losses, G_losses]).T, 
                columns = ["discriminator", "generator"]).to_csv(f"saved_models/{savefile}.csv", index=False)


    #Output
    with torch.no_grad():
        test_z = torch.randn(bs, z_dim).to(device)
        generated = G(test_z)

        save_image(generated.view(generated.size(0), 3, 224, 224), './generated_batch' + '.png')
