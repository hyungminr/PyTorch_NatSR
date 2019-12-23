import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchsummary import summary as torchsummary
from utils import update_lr, sec2time
from utils.data_loader import get_loader
from utils.train_NMD import *
from models.networks import Generator, Discriminator, NMDiscriminator
import time
from tensorboardX import SummaryWriter
import numpy as np

summary = SummaryWriter()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
eps = 1e-7
batch_size = 12
G = Generator().to(device)
D = Discriminator().to(device)
NMD = NMDiscriminator().to(device)


# torchsummary(G, input_size=(3, 112, 112))

# torchsummary(D, input_size=(3, 448, 448))

# torchsummary(NMD, input_size=(3, 448, 448))

NMD.load_state_dict(torch.load('./models/weights/NMD.pth'))

num_epochs = 1000

learning_rateG = 1e-4
learning_rateD = 1e-4

criterion_CE = nn.BCELoss()
criterion_L1 = nn.L1Loss()

optimizerG = torch.optim.Adam(G.parameters(), lr=learning_rateG)
optimizerD = torch.optim.Adam(D.parameters(), lr=learning_rateD)

loader = get_loader(batch_size=batch_size, train=True)

total_iter = len(loader)

_ = NMD.eval()

# data_iter = iter(loader)
# try:
#     lr, hr = next(data_iter)
# except StopIteration:
#     data_iter = iter(loader)
#     lr, hr = next(data_iter)

stime = time.time()
total_epoch_iter = total_iter * num_epochs
iter_count = 0
for epoch in range(num_epochs):
    for i, (lr, hr) in enumerate(loader):
        iter_count += 1
        
        lr = lr.to(device)
        hr = hr.to(device)

        sr = G(lr)

        Dhr = D(hr)
        Dsr = D(sr)

        target_real = torch.ones((sr.shape[0], 1), dtype=torch.float).to(device)
        target_fake = torch.zeros((sr.shape[0], 1), dtype=torch.float).to(device)

        loss_D = criterion_CE(Dsr, target_fake)
        loss_D += criterion_CE(Dhr, target_real)
        # loss_D = -torch.mean(torch.log(Dsr + eps)) - torch.mean(torch.log(1-Dhr+eps))

        optimizerD.zero_grad()
        loss_D.backward()
        optimizerD.step()
        
        summary.add_scalar(f'loss D/loss D', loss_D.data.cpu().numpy(), iter_count)

        sr = G(lr)
        nm = NMD(sr)
        Dhr = D(hr)
        Dsr = D(sr)
        loss_recon = criterion_L1(sr, hr)
        # loss_recon = torch.mean(torch.abs(hr - sr))
        loss_natural = criterion_CE(nm, target_real)
        # loss_natural = torch.mean(-torch.log(nm + eps))

        loss_G = criterion_CE(Dhr, target_fake)
        loss_G += criterion_CE(Dsr, target_real)
        # loss_G = -torch.mean(torch.log(Dhr + eps)) - torch.mean(torch.log(1-Dsr+eps))

        lambda_1 = 1
        lambda_2 = 1e-3
        lambda_3 = 1e-3
        loss_overall = lambda_1*loss_recon + lambda_2*loss_natural + lambda_2*loss_G

        optimizerG.zero_grad()
        loss_overall.backward()
        optimizerG.step()

        summary.add_scalar(f'loss G/loss recon', loss_recon.data.cpu().numpy(), iter_count)
        summary.add_scalar(f'loss G/loss natural', loss_natural.data.cpu().numpy(), iter_count)
        summary.add_scalar(f'loss G/loss G', loss_G.data.cpu().numpy(), iter_count)
        summary.add_scalar(f'loss G/loss Overall', loss_overall.data.cpu().numpy(), iter_count)

        etime = time.time() - stime
        rtime = etime * (total_epoch_iter-iter_count) / (iter_count+eps)
        print(f'Epoch: {epoch+1:03d}/{num_epochs:03d}, Iter: {i+1:04d}/{total_iter:04d}, ', end='')
        print(f'Loss G: {loss_overall.data:.4f}, Loss D: {loss_D.data:.4f}, ', end='')
        print(f'Elapsed: {sec2time(etime)}, Remaining: {sec2time(rtime)}')
        
        if (i+1) % 10 == 0:                        
            summary.add_image(f'image/sr_image', sr[0], iter_count) 
            summary.add_image(f'image/lr_image', lr[0], iter_count) 
            summary.add_image(f'image/hr_image', hr[0], iter_count)
        
    torch.save(G.state_dict(), f'./models/weights/G_epoch_{epoch+1}_loss_{loss_overall.data:.4f}.pth')
    torch.save(D.state_dict(), f'./models/weights/D_epoch_{epoch+1}_loss_{loss_D.data:.4f}.pth')
    
    if (epoch+1) % 10 == 0:
        learning_rateG *= 0.5
        learning_rateD *= 0.5
    
    update_lr(optimizerG, learning_rateG)
    update_lr(optimizerD, learning_rateD)

