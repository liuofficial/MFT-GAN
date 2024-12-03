import torch.nn.functional as F
from Network.Transformer import *

class downsampling(nn.Module):
    def __init__(self,in_c,out_c):
        super(downsampling, self).__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=in_c,out_channels=out_c,kernel_size=(3,3),padding=1),
            nn.LeakyReLU()
        )
    def forward(self,input):
        output = self.down(input)
        return output

class upsampling(nn.Module):
    def __init__(self,in_c,out_c):
        super(upsampling, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_c, out_channels=in_c, kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=in_c,out_channels=out_c,kernel_size=(3,3),padding=1),
            nn.LeakyReLU()
        )
    def forward(self,input):
        output = self.up(input)
        return output

class gloab_net(nn.Module):
    def __init__(self,hs_band,ms_band):
        super(gloab_net, self).__init__()
        inc1 = [64, 128, 256]
        inc2 = [32, 64, 128]

        self.up1 = upsampling(in_c=inc1[0], out_c=inc1[1])
        self.up2 = upsampling(in_c=inc1[1], out_c=inc1[2])


        self.down1 = downsampling(in_c=inc2[0], out_c=inc2[1])
        self.down2 = downsampling(in_c=inc2[1], out_c=inc2[2])

        self.CSAB1 = CSABlock(in_c1=inc1[0], in_c2=inc2[2])
        self.CSAB2 = CSABlock(in_c1=inc1[1], in_c2=inc2[1])
        self.CSAB3 = CSABlock(in_c1=inc1[2], in_c2=inc2[0])

        self.up3 = upsampling(in_c=inc1[0], out_c=inc1[1])
        self.up4 = upsampling(in_c=inc1[1], out_c=inc1[2])

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=hs_band, out_channels=inc1[0], kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=ms_band, out_channels=inc2[0], kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=inc1[2], out_channels=hs_band, kernel_size=(1, 1)),
        )

    def forward(self, y,z):
        y_up = F.interpolate(y, scale_factor=4, mode='bicubic', align_corners=False)
        y1 = self.conv1(y)
        # print("y1.shape:",y1.shape)
        y2 = self.up1(y1)
        # print("y2.shape:",y2.shape)
        y3 = self.up2(y2)
        # print("y3.shape:",y3.shape)
        z1 = self.conv2(z)
        # print("z1.shape",z1.shape)
        z2 = self.down1(z1)
        # print("z2.shape", z2.shape)
        z3 = self.down2(z2)
        # print("z3.shape", z3.shape)
        SA1 = self.CSAB1(y1,z3)
        # print("A1.shape:",SA1.shape)
        SA2 = self.CSAB2(y2,z2)
        # print("A2.shape:", SA2.shape)
        SA3 = self.CSAB3(y3,z1)
        # print("A3.shape:", SA3.shape)
        F1 = torch.add(self.up3(SA1), SA2)
        # print(F1.shape)
        F2 = torch.add(self.up4(F1), SA3)
        # F3 = torch.add(self.up5(F1), F2)
        out = self.conv3(F2) + y_up
        return out






