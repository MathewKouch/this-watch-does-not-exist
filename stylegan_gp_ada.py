import torch
import torch.optim as optim
import torchvision
from torchvision.transforms.functional import rotate as Rotate
from torchvision.datasets import ImageFolder
from torchmetrics.image.fid import FrechetInceptionDistance

import random # to generate fake labels
import pandas as pd
import torchvision.transforms as T

import os
import numpy as np
from IPython.display import clear_output
import imageio
from tqdm import tqdm
import imageio
import os
os.cpu_count()
from util import set_seed, calc_gradient_penalty, AdaptiveAugmenter, step
from model import discriminator, g_stylegan1
import wandb
torch.cuda.empty_cache()
set_seed(42)

# Hyper prams
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'   
print(f'training on {device}') 

WITH_CLIPPING = False 

CP = True # Run model from checkpoint
CONT = 'cont_' if CP else '' # Continuation flag for wandb RUN_NAME
dlr = 5e-5 if WITH_CLIPPING else 1e-4 
glr = 5e-5 if WITH_CLIPPING else 1e-4
betas = None if WITH_CLIPPING else (0., 0.999) 

img_size =  256
d_iterations = 2 # train discriminator more than generator
batch_size = 16 if WITH_CLIPPING else 64 #max 64,  fp32: 64 max, fp16: 
latent_noise_dem = 256 #128
C = 0.01 # weight clipping -C to C
FID_ITER = 5 # epochs to FID
ADA_PROB = 0.1 # inital ada aug prob
G_ch = 64 # Generatorr emb dim
D_ch = 64 # Discriminator emb dimension
n = 1
n_cores = 13
START_EPOCH = 0
MAX_EPOCHS = 1000

# regularisation
lambda_gp = 10
SOFT_LABELS = True

gamma = 0.05 #0.01 # Initial decaying Input noise scalar, 0.1 too big
alpha = 0.95 # exponential decay rate, 0.99 decays too slow. 0.9 shows no noise after 30 epochs
CLASSES = np.arange(2)

# FID
fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device) # Features 64, 192, 768, 2048
BEST_FID = 9999999

# Data
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]
real_transform = T.Compose([
            T.Resize(img_size),
            T.CenterCrop((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std)
            ])

mean = torch.tensor(mean).reshape(3,1,1)
std = torch.tensor(std).reshape(3,1,1)

trainset = ImageFolder('watch_images_4', transform=real_transform)

train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=n_cores,
        shuffle=True,
        drop_last=True)
#fixed_latent_noise = torch.randn(16, latent_noise_dem, n, n).to(device)
fixed_latent_noise = torch.randn(16, 1, 1, latent_noise_dem).to(device)
print(f'There are {len(trainset)} samples in dataset')

# Models

G = g_stylegan1(latent_noise_dem, ch=G_ch, img_size=img_size, device=device).to(device)
D = discriminator(D_ch).to(device)

G_optimizer = optim.RMSprop(G.parameters(), lr=glr) if WITH_CLIPPING else \
                optim.Adam(G.parameters(), lr=glr, betas=betas)

D_optimizer = optim.RMSprop(D.parameters(), lr=dlr) if WITH_CLIPPING else \
                optim.Adam(D.parameters(), lr=dlr, betas=betas)

#scaler = torch.cuda.amp.GradScaler() # for mixed precision (increasing batch size =))
ada = AdaptiveAugmenter(p=ADA_PROB, device=device)

if CP:
    cp = torch.load('256_wganloss_gp_ADA.pt', map_location=device)

    G.load_state_dict(cp['generator'])
    D.load_state_dict(cp['discriminator'])

    G_optimizer.load_state_dict(cp['g_optim'])
    D_optimizer.load_state_dict(cp['d_optim'])
    if 'ada' in cp.keys():
        ada = cp['ada']

    START_EPOCH = cp['epochs']
        
G_params = sum(p.numel() for p in G.parameters())
D_params = sum(p.numel() for p in D.parameters())
print(f'Generator has {G_params} params. Discriminator has {D_params} params')

D_losses = []
D_out_real = []
D_out_fake = []
G_losses = []
test_images_log = []
total_images_seen = 0 if CP==False else cp['epochs']*17691

wandb.init(project="DCGANs",
          name = f'{CONT}stylegan-ADA_{img_size}_{latent_noise_dem}D_{D_ch}Dembd_{G_ch}Gembd',
          config={
              'G_name': G.__class__.__name__,
              'D_name': D.__class__.__name__,
              'G_params': G_params,
              'D_params': D_params,
              'D_ch': D_ch,
              'G_ch': G_ch,
              'c': C, # D weight clipping
              "dlr": dlr,
              "batch_size":         batch_size,
              "epochs":           MAX_EPOCHS,
              'D_optim':D_optimizer.__class__.__name__,
              'G_optim':G_optimizer.__class__.__name__,
              'loss': 'wloss_with_grad_penalty',
              'gradient_penalty_weight': lambda_gp, # gradient penalty weight
              'img_size': img_size,
              'latent_dim': latent_noise_dem,
              'num_workers': n_cores,
              'd_iter': d_iterations, # number of times to train discr per batch
          })




real_images = next(iter(train_loader))[0][:16].mul(std).add(mean)

grid = (torchvision.utils.make_grid(real_images, nrow=4).permute(1,2,0).numpy()*255).astype(np.uint8)
wandb.log({
    'epoch':0,
    'real_image_samples':wandb.Image(grid),
})

for epoch in range(START_EPOCH, MAX_EPOCHS):
    
    loader = tqdm(train_loader) 
    
    G.train()
    D.train()
    
    D_losses = 0
    G_losses = 0
    
    D_real_acc, D_fake_acc = 0, 0
    
    count = 0
    iterations = 0
    for num_iter, images in enumerate(loader):

        mini_batch = len(images[0]) # number of images
        
        images = images[0].to(device)
        
        if epoch%FID_ITER==0:
            fid.update(images, real=True)
            
        images = ada(images, True)# Augment images based off p
                
        ########### Train Discriminator D! ############
        for _ in range(d_iterations):
            D_iteration_loss = 0


            latent_noise = torch.randn(mini_batch, n, n, latent_noise_dem, device=device)
            fakes = G(latent_noise)
            fakes = ada(fakes, True)

            D_real = D(images).reshape(-1)
            D_fake = D(fakes).reshape(-1)

            D_loss = (
                -(torch.mean(D_real) - torch.mean(D_fake))
            )
            
            if WITH_CLIPPING==False:
                gp = calc_gradient_penalty(D, images, fakes, device=device, amp=False)
                D_loss = D_loss + lambda_gp*gp 
            
                    
            D.zero_grad()
            D_loss.backward(retain_graph=True) # retain graph = True needed to train generator below             
            D_optimizer.step()
            
            if WITH_CLIPPING:
                for p in D.parameters():
                    p.data.clamp_(-C,C)

            D_iteration_loss += D_loss.item()
                        
            with torch.no_grad():
                
                D_real_acc += torch.mean(step(D_real.detach()))
                
                D_fake_acc += (1-torch.mean(step(D_fake.detach())))
                
                iterations += 1
                
                ada.update(D_real.detach()) # update p with D predictions of real image logits
            
                
        ########### Train Generator G ##############

        fakes = G(latent_noise)
        fakes = ada(fakes, True)
        output = D(fakes).reshape(-1)
        
        G_loss = -torch.mean(output)
        
        G.zero_grad()
        G_loss.backward()    
        G_optimizer.step()
        
        if epoch%FID_ITER==0:
            fakes_for_fid = fakes.detach()#.to(torch.uint8)
            fid.update(fakes_for_fid, real=False)

        del latent_noise
        del fakes
        G_losses += (G_loss.item())
        D_losses += (D_iteration_loss/d_iterations)

    # End of Epoch
        total_images_seen += mini_batch
        with torch.no_grad():
            if num_iter%100==0:
                test_fake = G(fixed_latent_noise).cpu().detach()
                test_fake = test_fake.mul(std).add(mean)
                imgs_np = (torchvision.utils.make_grid(test_fake, nrow=4, pad_value = 0.5).numpy().transpose((1, 2, 0))*255).astype(np.uint8)
                wandb.log({
                    'g_images_train':wandb.Image(imgs_np)
                })
                del test_fake
                del imgs_np
            
    #log the output of the generator given the fixed latent noise vector
    #with torch.cuda.amp.autocast():
    test_fake = G(fixed_latent_noise)
    test_fake = test_fake.cpu().detach()
    test_fake = test_fake.mul(std).add(mean)
    imgs_np = (torchvision.utils.make_grid(test_fake, nrow=4, pad_value = 0.5).numpy().transpose((1, 2, 0))*255).astype(np.uint8)

    test_images_log.append(imgs_np)
    
    if epoch%FID_ITER==0:
        FID = fid.compute()
        fid.reset()
    
    D_avg_loss = D_losses/len(loader)
    G_avg_loss = G_losses/len(loader)

    wandb.log({
        'epoch': epoch,
        'ada_p':ada.probability,
        'fid': FID,
        'g_images': wandb.Image(imgs_np),
        'D_loss': D_avg_loss,
        'D_real_acc': D_real_acc/iterations,
        'D_fake_acc': D_fake_acc/iterations,
        'G_loss': G_avg_loss,
        'total_images_seen': total_images_seen,
    })

    if total_images_seen<1000:
        total_images_seen_show = total_images_seen
    else:
        total_images_seen_show = total_images_seen/1000
    print(f'epoch {epoch+1}/{MAX_EPOCHS} | FID: {FID} | D_Loss: {D_avg_loss:.4f} | G_loss: {G_avg_loss:.4f} | kimg: {total_images_seen_show:.1f}')
    
    checkpoint = {
        'generator': G.state_dict(),
        'discriminator': D.state_dict(),
        'g_optim': G_optimizer.state_dict(),
        'd_optim': D_optimizer.state_dict(),
        'mean': mean,
        'std': std,
        'image_size': img_size,
        'test_logs': test_images_log,
        'latent_noise_dem': latent_noise_dem,
        'd_losses': D_losses,
        'g_losses': G_losses,
        'ada': ada,
        'epochs':epoch,
        'max_epochs': MAX_EPOCHS,
        }

    cp_name = f'{img_size}_wganloss_clipping_ADA_latest.pt' if WITH_CLIPPING else f'{img_size}_wganloss_gp_ADA_latest.pt'
    torch.save(checkpoint, cp_name)

    if FID<BEST_FID:
        BEST_FID=FID
        
        cp_name = f'{img_size}_wganloss_clipping_ADA.pt' if WITH_CLIPPING else f'{img_size}_wganloss_gp_ADA.pt'
        torch.save(checkpoint, cp_name)
        imageio.mimsave(f'{img_size}_{latent_noise_dem}D.gif', test_images_log)

    del checkpoint
    del test_fake
    del imgs_np