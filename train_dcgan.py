import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
from dataload import *
from gan_architecture import *
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import get_n_params, accuracy, add_noise

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bs = 128
batch_size_test = 128

train_loader = define_loaders(bs, batch_size_test, 
                                    rescale=256,
                                    crop=224,
                                    rgb=True,
                                    test_set=False)

lr = 0.00008
beta1 = 0.5
n_epoch = 3000

# build network
z_dim = 1024
labels = torch.full((bs,1), 1.0, dtype=torch.float, device=device)
# landscape_dim = 224*224

G = Generator(z_dim).to(device)
D = Discriminator().to(device)

# print(G)
# print(D)

# loss
criterion = nn.BCELoss() 

# optimizer
G_optimizer = optim.Adam(G.parameters(), lr = lr, betas=(beta1, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr = lr, betas=(beta1, 0.999))


def D_train(x):#, labels):
    #=======================Train the discriminator=======================#
    D.zero_grad()

    # train discriminator on real
    x_real = x.view(-1, 3, 224 , 224).to(device)
    size = len(x_real)

    D_output = D(x_real)

    if size != bs:
        y_real = torch.ones(size, 1).to(device)
        D_real_acc = accuracy(D_output, y_real)
        D_real_loss = criterion(D_output, y_real)
    else:
        labels.fill_(1.0)
        D_real_acc = accuracy(D_output, labels)
        D_real_loss = criterion(D_output, labels)


    D_real_loss.backward()

    # train discriminator on fake
    z = torch.randn(size, z_dim).to(device) 
    with torch.no_grad():
        #x_fake, y_fake = G(z), torch.zeros(size, 1).to(device)
        x_fake = G(z)
    labels.fill_(0.0)

    D_output = D(x_fake)
    if size != bs:
        D_fake_loss = criterion(D_output, y_real)
        D_fake_acc = accuracy(D_output, y_real)
    else:
        D_fake_loss = criterion(D_output, labels)
        D_fake_acc = accuracy(D_output, labels)

    D_fake_loss.backward()
    # gradient backprop & optimize ONLY D's parameters
    full_loss = D_real_loss + D_fake_loss
    #D_loss.backward()
    D_optimizer.step()

    return full_loss.data.item(), ((D_real_acc + D_fake_acc) / 2).item()

def G_train(x):#, labels):
    #=======================Train the generator=======================#
    G.zero_grad()

    z = torch.randn(bs, z_dim).to(device)
    #y = torch.ones(bs, 1).to(device)
    labels.fill_(1.0)

    G_output = G(z)
    D_output = D(G_output)
    G_loss = criterion(D_output, labels)
    G_acc = accuracy(D_output, labels)

    # gradient backprop & optimize ONLY G's parameters
    G_loss.backward()
    G_optimizer.step()
    

    return G_loss.data.item(), G_acc.item()

if __name__ == "__main__":
    D_losses, G_losses = [0], [0]
    D_accs, G_accs = [0], [0]
    savefile = 'res-gan-2'
    k = 2 # Facteur d'apprentissage discriminateur

    print(f'Launching for {n_epoch} epochs...')
    print(f"Number of parameters : D : {get_n_params(D)}, G : {get_n_params(G)}")
    
    for epoch in range(1, n_epoch+1):

        for batch_idx, (x) in enumerate(tqdm(train_loader)):

            if(len(x) < bs) :
                labels = torch.full((len(x),1), 1.0, dtype=torch.float, device=device)
            elif(len(x) == bs and len(labels) < bs):
                labels = torch.full((bs,1), 1.0, dtype=torch.float, device=device)

            # if(len(x) == bs) :
            D_current_loss, D_current_acc = D_train(x)
            # else:
            #     labels_cpu = torch.full((len(x),1), 1.0, dtype=torch.float, device=device)
            #     D_current_loss, D_current_acc = D_train(x, labels_cpu)

            if batch_idx % k == 0:
                G_current_loss, G_current_acc = G_train(x)
            else:
                G_current_loss = G_losses[-1]
                G_current_acc = G_accs[-1]


            if ((batch_idx % 2 == 0) & (batch_idx > 0)) | (batch_idx == len(train_loader)-1):
                D_losses.append(D_current_loss)
                G_losses.append(G_current_loss)
                D_accs.append(D_current_acc)
                G_accs.append(G_current_acc)


                

        print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % (
                (epoch), n_epoch, np.mean(D_losses[-10:]), np.mean(G_losses[-10:])))

        if (savefile is not None) and (epoch % 50 == 0) and (epoch > 0):
            torch.save(G.state_dict(), f"saved_models/{savefile}_generator.sav")
            torch.save(D.state_dict(), f"saved_models/{savefile}_discriminator.sav")
            pd.DataFrame(data=np.array([D_losses, G_losses]).T, 
                columns = ["discriminator", "generator"]).to_csv(f"saved_models/{savefile}_losses.csv", index=False)
            pd.DataFrame(data=np.array([D_accs, G_accs]).T, 
                columns = ["discriminator", "generator"]).to_csv(f"saved_models/{savefile}_accs.csv", index=False)

        if(epoch%100 == 0):
            # print("Saving epoch " + str(epoch))
            with torch.no_grad():
                test_z = torch.randn(4, z_dim).to(device)
                generated = G(test_z)

                save_image(generated.view(generated.size(0), 3, 224, 224), './generated_batchs/generated_batch' + str(epoch) + '.png')
