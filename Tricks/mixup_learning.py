import numpy as np
import torch

def mixup_learning(alpha, images, target, model, criterion):
    lamda = np.random.beta(alpha, alpha)
    index = torch.randperm(images.size(0)).cuda()
    inputs = lamda * images + (1 - lamda) * images[index, :]
    target_a, target_b = target, target[index]
    outputs = model(inputs)
    loss = lamda * criterion(outputs, target_a) + (1 - lamda) * criterion(outputs, target_b)
    return loss