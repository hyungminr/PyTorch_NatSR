import torch
import torch.nn as nn
from utils.data_loader import get_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def data_gen_NMD(lr, hr, alpha=0.5, sigma=0.1):
    upsample = nn.Upsample(scale_factor=4)
    up = upsample(lr)    
    noise = torch.randn_like(hr) * sigma    
    A = alpha * hr + (1-alpha) * up
    B = hr + noise    
    return A, B

def data_gen_alpha(lr, hr, alpha=0.5):
    upsample = nn.Upsample(scale_factor=4)
    up = upsample(lr)    
    A = alpha * hr + (1-alpha) * up
    return A

def data_gen_sigma(lr, hr, sigma=0.1):
    noise = torch.randn_like(hr) * sigma    
    B = hr + noise    
    return B

"""
def evaluate_NMD(NMD, alpha=0.5, sigma=0.1):
    NMD.eval()
    data_loader = get_loader(train=False, batch_size=1)
    data_iter = iter(data_loader)
    flag = True
    right = 0
    count = 0
    for _ in range(10):
        try:
            lr, hr = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            lr, hr = next(data_iter)

        lr = lr.to(device)
        hr = hr.to(device)
        A, B = data_gen_NMD(lr, hr, alpha, sigma)
        if NMD(hr)[:,0] > 0.5:
            right += 1
        if NMD(A)[:,0] < 0.5:
            right += 1
        if NMD(B)[:,0] < 0.5:
            right += 1
        count += 3
    return right / count
"""

def evaluate_NMD(NMD, param=0.5, mode='alpha'):
    NMD.eval()
    data_loader = get_loader(train=False, batch_size=1)
    data_iter = iter(data_loader)
    flag = True
    right = 0
    count = 0
    for _ in range(25):
        try:
            lr, hr = next(data_iter)
        except StopIteration:
            data_iter = iter(data_loader)
            lr, hr = next(data_iter)
        lr = lr.to(device)
        hr = hr.to(device)
        
        if NMD(hr)[:,0] > 0.5:
            right += 1
            
        if mode == 'alpha':
            A = data_gen_alpha(lr, hr, param)
            if NMD(A)[:,0] < 0.5:
                right += 1
                
        elif mode == 'sigma':
            B = data_gen_sigma(lr, hr, param)
            if NMD(B)[:,0] < 0.5:
                right += 1
                
        else:
            print(f'wrong mode: {mode}. It should be alpha or sigma')
            assert False
            
        count += 2
        
    return right / count
