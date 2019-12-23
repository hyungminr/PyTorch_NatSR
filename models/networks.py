import os
import torch
import torch.nn as nn
    
class ConvLayer(nn.Module):
    def __init__(self, dim_in, dim_out, kernel=3, stride=1, padding=1, bias=True):
        super(ConvLayer, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_channels=dim_in, out_channels=dim_out, kernel_size=kernel, stride=stride, padding=padding, bias=bias)]
        layers += [nn.ReLU(inplace=True)]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
        
class RDBlock(nn.Module):
    def __init__(self, dim=64, kernel=3, stride=1, padding=1, bias=True):
        super(RDBlock, self).__init__()        
        self.conv1 = ConvLayer(dim, dim, kernel, stride, padding, bias)
        self.conv2 = ConvLayer(dim*2, dim, kernel, stride, padding, bias)
        self.conv3 = ConvLayer(dim*3, dim, kernel, stride, padding, bias)   
        self.conv4 = ConvLayer(dim*4, dim, kernel, stride, padding, bias)   
        self.conv5 = ConvLayer(dim*5, dim, kernel, stride, padding, bias)   
        self.conv6 = ConvLayer(dim*6, dim, kernel, stride, padding, bias)        
        layer = [nn.Conv2d(in_channels=dim*7, out_channels=dim, kernel_size=1, stride=1, padding=0, bias=bias)]
        self.conv_out = nn.Sequential(*layer)
        
    def forward(self, x):
        c1 = self.conv1(x) # 64 to 64
        x1 = torch.cat([x, c1], dim=1) # 64 + 64 = 128
        c2 = self.conv2(x1) # 128 to 64
        x2 = torch.cat([x, c1, c2], dim=1) # 64 + 64 + 64 = 192
        c3 = self.conv3(x2) # 192 to 64
        x3 = torch.cat([x, c1, c2, c3], dim=1) # 
        c4 = self.conv4(x3) # 
        x4 = torch.cat([x, c1, c2, c3, c4], dim=1) # 
        c5 = self.conv5(x4) # 
        x5 = torch.cat([x, c1, c2, c3, c4, c5], dim=1) # 
        c6 = self.conv6(x5) # 
        x6 = torch.cat([x, c1, c2, c3, c4, c5, c6], dim=1) # 64 * 7
        x7 = self.conv_out(x6) # 64*7 to 64
        return torch.add(x, x7*0.1)

class UpsampleBlock(nn.Module):
    
    def __init__(self, dim_in=64, scale=2, bias=True):
        super(UpsampleBlock, self).__init__()
        layers = []
        layers += [nn.Conv2d(in_channels=dim_in, out_channels=dim_in*4, kernel_size=3, stride=1, padding=1, bias=bias)]
        layers += [nn.PixelShuffle(upscale_factor=scale)]
        self.Upsample = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.Upsample(x)
    
def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class Generator(nn.Module):
    def __init__(self, bias=True):
        super(Generator, self).__init__()
        
        layers = []
        layers += [nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=bias)]
        # layers += [nn.ReLU(inplace=True)]
        self.conv_in = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=bias)]
        # layers += [nn.ReLU(inplace=True)]
        self.conv_out = nn.Sequential(*layers)
        
        # self.sub = nn.PixelShuffle(upscale_factor=2)
        
        layers = []
        layers += [nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=bias)]
        layers += [nn.Hardtanh(min_val=0, max_val=1.0)]
        self.o = nn.Sequential(*layers)
        
        self.conv_up_1 = UpsampleBlock()
        self.conv_up_2 = UpsampleBlock()
                
        self.rdb1_1 = RDBlock()
        self.rdb1_2 = RDBlock()
        self.rdb2_1 = RDBlock()
        self.rdb2_2 = RDBlock()
        self.rdb3_1 = RDBlock()
        self.rdb3_2 = RDBlock()
        self.rdb4_1 = RDBlock()
        self.rdb4_2 = RDBlock()
                
    def torch_add(self, add_list):
        x = add_list[0]
        for a in add_list[1:]:
            x = torch.add(x, a)
        return x
        
    def forward(self, x):
        x0 = self.conv_in(x)
        
        x1 = self.rdb1_1(x0)
        x1 = self.rdb1_2(x1)
        
        x1 = self.torch_add([x0, x1])
        
        x2 = self.rdb2_1(x1)
        x2 = self.rdb2_2(x2)
        
        x2 = self.torch_add([x0, x1, x2])
        
        x3 = self.rdb3_1(x2)
        x3 = self.rdb3_2(x3)
        
        x3 = self.torch_add([x2, x3])
        
        x4 = self.rdb4_1(x2)
        x4 = self.rdb4_2(x4)
        
        x4 = self.torch_add([x0, x2, x3, x4])
        
        x5 = self.conv_out(x4)
        
        x = self.torch_add([x0, x5]) # 64
        
        x = self.conv_up_1(x)
        x = self.conv_up_2(x)
        
        x = self.o(x)
        
        return x

class Discriminator(nn.Module):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1_1 = nn.Conv2d(in_channels=3,  out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)        
        
        self.conv2_1 = nn.Conv2d(in_channels=64,  out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.conv5 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        
        layers = []
        layers += [nn.utils.spectral_norm(self.conv1_1)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.utils.spectral_norm(self.conv1_2)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        layers += [nn.utils.spectral_norm(self.conv2_1)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.utils.spectral_norm(self.conv2_2)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        layers += [nn.utils.spectral_norm(self.conv3_1)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.utils.spectral_norm(self.conv3_2)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        layers += [nn.utils.spectral_norm(self.conv4_1)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.utils.spectral_norm(self.conv4_2)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        layers += [nn.utils.spectral_norm(self.conv5)]
        
        self.cnn = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.Linear(28 * 28, 1)]
        layers += [nn.Sigmoid()]
        
        self.fc = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)        
        return out

class NMDiscriminator(nn.Module):
    
    def __init__(self):
        super(NMDiscriminator, self).__init__()
        
        layers = []
        layers += [nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        layers += [nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        layers += [nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        layers += [nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)]
        layers += [nn.ReLU(inplace=True)]
        layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        
        layers += [nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)]
                
        self.cnn = nn.Sequential(*layers)
        
        layers = []
        layers += [nn.Linear(28 * 28, 1)]
        layers += [nn.Sigmoid()]
        
        self.fc = nn.Sequential(*layers)
        
    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)        
        return out
