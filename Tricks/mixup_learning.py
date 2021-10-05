import numpy as np
import torch

def mixup_learning(alpha, images, target, model, criterion):
    lamda = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0)).cuda()
    inputs = lamda * images + (1 - lamda) * images[index, :]
    target_mixup = lamda * target + (1 - lamda) * target[index]
    outputs = model(inputs) 
    loss = criterion(outputs, target_mixup)
    return loss
