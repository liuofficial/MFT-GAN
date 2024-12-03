import torch
from Network import discriminator, generator
import torch.nn as nn
from thop import profile
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class G_net(nn.Module):
    def __init__(self,hs_band, pan_band):
        super(G_net, self).__init__()
        self.NET = generator.gloab_net(hs_band, pan_band)
    def forward(self, Y,Z):
        out = self.NET(Y,Z)
        return out




class D1_net(torch.nn.Module):
    def __init__(self, hs_band):
        super().__init__()
        self.net = discriminator.spectrum_discriminator(in_c=hs_band, num_classes=1, H=32, depth=3, heads=4, dim_head=16, emb_dropout=0.)
        # H=48 for LA datasets
    def forward(self, x):
        out = self.net(x)
        return out


class D2_net(torch.nn.Module):
    def __init__(self, pan_band):
        super().__init__()
        self.net = discriminator.spatial_discriminator(in_c=pan_band, num_classes=1, heads=4, head_dim=16)
    def forward(self, x):
        out = self.net(x)
        return out


if __name__ == '__main__':
    
    D1_net = D1_net(31).to(device)
    D2_net = D2_net(1).to(device)
    G_net = G_net(31, 1).to(device)
    
    input1 = torch.randn((1, 31, 8, 8)).to(device)

    input1_up = torch.randn((1, 31, 32, 32)).to(device)
    input2 = torch.randn((1, 1, 32, 32)).to(device)
    input3 = torch.randn((1, 31, 32, 32)).to(device)
    
    Flops1, params1 = profile(D1_net, inputs=(input1_up,))  # macs
    print('Flops: % .4fG' % (Flops1 / 1000000000))  # 计算量
    print('params参数量: % .4fM' % (params1 / 1000000))
    Flops2, params2 = profile(D2_net, inputs=(input2,))  # macs
    print('Flops: % .4fG' % (Flops2 / 1000000000))  # 计算量
    print('params参数量: % .4fM' % (params2 / 1000000))
    
    Flops3, params3 = profile(G_net, inputs=(input1, input2,))  # macs
    print('Flops: % .4fG' % (Flops3 / 1000000000))  # 计算量
    print('params参数量: % .4fM' % (params3 / 1000000))
    

    print('Total Flops: % .4fG' % ((Flops1 + Flops2 + Flops3) / 1000000000))  # 计算量
    print('Total params参数量: % .4fM' % ((params1 + params2 + params3) / 1000000))  # 参数量：等价与上面的summary输出的Total params值





