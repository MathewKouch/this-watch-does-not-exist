import os
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
import torchvision
import torchvision.utils as vutils
import random # to generate fake labels
import pandas as pd
import torchvision.transforms as T
from torch.utils.data import Dataset
#import albumentations as A
#mport cv2

import os
import numpy as np
from PIL import Image

def set_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

class DS(Dataset):
    def __init__(self, ROOT, csv, transforms=None):
        self.ROOT = ROOT
        self.df = pd.read_csv(csv)
        self.tf = transforms
        
        self.image_files = self.df.image_name
        self.prices = self.df.price
        self.brands = self.df.brand
        self.names = self.df.name
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        img_name = self.df.image_name.iloc[idx]
        img = Image.open(os.path.join(self.ROOT, 'images', img_name))

        if self.tf!=None:
            img = self.tf(img)
            
        price = self.prices.iloc[idx]
        brand = self.brands.iloc[idx]
        name = self.names.iloc[idx]
        
        return img
    
class DS_images(Dataset):
    def __init__(self, ROOT, image_folder, transforms=None):
        self.ROOT = ROOT
        self.images_unfiltered = os.listdir(os.path.join(ROOT, image_folder)) # list of images

        self.images = []
        for fn in self.images_unfiltered:
            if fn.endswith('.jpg'):
                self.images.append(fn)
      
        self.tf = transforms
        self.image_folder = image_folder
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        img = Image.open(os.path.join(self.ROOT, self.image_folder, img_name))

        if self.tf!=None:
            img = self.tf(img)
        
        return img

def calc_gradient_penalty(critic, real, fake, device='cpu', amp=False):

    bs, c, h, w = real.shape
    if amp:
        with torch.cuda.amp.autocast():
            epsilon = torch.rand((bs, 1, 1, 1), device=device).repeat(1, c, h, w).half()
            interpolated_images = real*epsilon + fake*(1-epsilon)

            mixed_scores = critic(interpolated_images)

            # gradient of mixed scores wrt to interpolated images
            gradient = torch.autograd.grad(
                inputs=interpolated_images,
                outputs=mixed_scores,
                grad_outputs=torch.ones_like(mixed_scores),
                create_graph=True,
                retain_graph=True,
            )[0]

            # Calc gradient penalty
            gradient = gradient.view(gradient.shape[0], -1)
            gradient_norm = gradient.norm(2, dim=1)
            # return gradient penalty
            return torch.mean((gradient_norm - 1) ** 2)
    else:
        epsilon = torch.rand((bs, 1, 1, 1), device=device).repeat(1, c, h, w)
        interpolated_images = real*epsilon + fake*(1-epsilon)

        mixed_scores = critic(interpolated_images)

        # gradient of mixed scores wrt to interpolated images
        gradient = torch.autograd.grad(
            inputs=interpolated_images,
            outputs=mixed_scores,
            grad_outputs=torch.ones_like(mixed_scores),
            create_graph=True,
            retain_graph=True,
        )[0]

        # Calc gradient penalty
        gradient = gradient.view(gradient.shape[0], -1)
        gradient_norm = gradient.norm(2, dim=1)
        # return gradient penalty
        return torch.mean((gradient_norm - 1) ** 2)


    from torch.autograd import Variable
import torchvision.transforms as T

def step(logits):
    'values is a tensor of logits'
    # negative values -> 0.0, positive values -> 1.0
    return 0.5 * (1.0 + torch.sign(logits))


class AdaptiveAugmenter(nn.Module):

    def __init__(self, p=0.0, target_accuracy=0.85, integration_steps=1000, device=None):
        super().__init__()
        
        self.device = device
        self.target_accuracy = target_accuracy
        self.integration_steps = integration_steps
        
        # stores the current probability of an image being augmented
        self.probability = torch.tensor([p], device=device)
        
        self.augmenter = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomApply(nn.ModuleList([T.RandomRotation((90,90)),]), p=0.5),
                T.RandomApply(nn.ModuleList([T.RandomRotation((-90,-90)),]), p=0.5),
                T.RandomApply(nn.ModuleList([T.RandomRotation((180,180)),]), p=0.1), 
            ],
        )
        

    # "hard sigmoid", useful for binary accuracy calculation from logits
    def step(self, values):
        'values is a tensor of logits'
        # negative values -> 0.0, positive values -> 1.0
        return 0.5 * (1.0 + torch.sign(values))

    def forward(self, images, training):
        if training:
            bs = images.shape[0]
            #images = images.permute(0,2,3,1).numpy()
            #print(images.shape)
#             augmented_images = [
#                 torch.tensor(self.augmenter(image=images[i])['image']) for i in range(bs)
#             ]
            augmented_images = self.augmenter(images)
            # during training either the original or the augmented images are selected
            # based on self.probability
#             augmentation_values = tf.random.uniform(
#                 shape=(batch_size, 1, 1, 1), minval=0.0, maxval=1.0
#             )
            augmentation_values = torch.rand((bs, 1, 1, 1), device=self.device)
    
            #augmentation_bools = tf.math.less(augmentation_values, self.probability)
            augmentation_bools = torch.less(augmentation_values, self.probability)
            #print(augmentation_bools.reshape(4,4))
            #images = tf.where(augmentation_bools, augmented_images, images)
            
            # COnvert back to torch tensor with dim B, C, H, W
            images = torch.where(augmentation_bools, augmented_images, images)
            
            del augmentation_values
            del augmentation_bools
            del augmented_images
            
        return images

    def update(self, real_logits):
        ''' logits are 2D with B x 1, a binary prediction for each image in B'''
        current_accuracy = torch.mean(self.step(real_logits))
        
        #print(current_accuracy.item())
        # the augmentation probability is updated based on the dicriminator's
        # accuracy on real images
        accuracy_error = current_accuracy - self.target_accuracy
        self.probability = self.probability + (accuracy_error / self.integration_steps)
            
        
    
        #print(self.probability.item())
        self.probability = torch.clip(
                self.probability, min=0, max=1
            
        )
