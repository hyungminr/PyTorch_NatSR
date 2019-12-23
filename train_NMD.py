import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
# from torchsummary import summary as torchsummary
from utils import update_lr, sec2time
from utils.data_loader import get_loader
from utils.train_NMD import *
from models.networks import NMDiscriminator
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = NMDiscriminator().to(device)

# torchsummary(model, input_size=(3, 448, 448))

model.load_state_dict(torch.load('./models/weights/NMD'))

# Hyper-parameters
num_epochs = 1000
learning_rate = 1e-4
eps = 1e-7

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

loader = get_loader(batch_size=24)

total_iter = len(loader)
current_lr = learning_rate

alpha = 0.5
sigma = 0.1

stime = time.time()
total_epoch_iter = total_iter * num_epochs

iter_count = 0
for epoch in range(num_epochs):
    for i, (lr, hr) in enumerate(loader):
        iter_count += 1
        lr = lr.to(device)
        hr = hr.to(device)
        A, B = data_gen_NMD(lr, hr, alpha, sigma)
        
        output_hr = model(hr)
        output_A = model(A)
        output_B = model(B)
        
        batch_size = lr.shape[0]
        
        target_real = torch.ones((batch_size, 1), dtype=torch.float).to(device)
        target_fake = torch.zeros((batch_size, 1), dtype=torch.float).to(device)
        
        loss_NMD1 = criterion(output_hr, target_real)
        loss_NMD2 = criterion(output_A,  target_fake)
        loss_NMD3 = criterion(output_B,  target_fake)

        loss = loss_NMD1 + loss_NMD2*0.1 + loss_NMD3
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            acc_alpha = evaluate_NMD(model, alpha, 'alpha')
            acc_sigma = evaluate_NMD(model, sigma, 'sigma')
            
            if alpha < 0.8 and acc_alpha >= 0.95:
                alpha += 0.1
                torch.save(model.state_dict(), f'./models/weights/NMD_alpha_{alpha:.1f}_sigma_{sigma:.4f}.pth')
                
            if sigma > 0.0044 and acc_sigma >= 0.95:
                sigma *= 0.8
                torch.save(model.state_dict(), f'./models/weights/NMD_alpha_{alpha:.1f}_sigma_{sigma:.4f}.pth')

            etime = time.time() - stime
            rtime = etime * (total_epoch_iter-iter_count) / (iter_count+eps)
            print(f'Epoch: {epoch+1:04d}/{num_epochs}, Iter: {i+1:03d}/{total_iter}, Loss: {loss.data:.4f},', end=' ')
            print(f'Acc alpha: {acc_alpha:.2f}, alpha: {alpha:.1f}, Acc sigma: {acc_sigma:.2f}, sigma: {sigma:.4f},', end=' ')
            print(f'Elapsed: {sec2time(etime)}, Remaining: {sec2time(rtime)}')
            
        if alpha >= 0.8 and sigma <= 0.0044:
            torch.save(model.state_dict(), f'./models/weights/NMD.pth')
            break
            
    if (epoch+1) % 10 == 0:
        current_lr *= 0.5
        update_lr(optimizer, current_lr)
