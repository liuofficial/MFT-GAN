from Network.Transformer import *
from einops import rearrange

class spectrum_discriminator(nn.Module):
    def __init__(self, in_c, num_classes, H, depth, heads, dim_head, emb_dropout=0.):
        super().__init__()
        out_c = 256
        self.Embedding = nn.Sequential(
            nn.Linear(H**2, out_c)
        )
        self.H = H
        # self.cls_token = nn.Parameter(torch.randn(1, 1, out_c))
        self.dropout = nn.Dropout(emb_dropout)
        self.TransformerE = Spetral_Transformer_Block(out_c, depth, heads, dim_head, 0.0)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(in_c),
            nn.Linear(in_c, num_classes)
        )
    def forward(self,X):
        B = X.size(0)
        E = rearrange(X, 'B c H W -> B c (H W)',H=self.H)
        E = self.Embedding(E)
        input1 = self.dropout(E)
        # print(E.shape)
        out1 = self.TransformerE(input1)
        # print(out1.shape)
        out1 = torch.mean(out1,dim=-1)
        # print(out1.shape)
        out2 = self.mlp_head(out1)
        out = torch.clip_(out2, -1., 1.)

        return out


class spatial_discriminator(nn.Module):
    def __init__(self,in_c,num_classes,heads,head_dim):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_c,out_channels=32, kernel_size=(3, 3),padding=1),
            nn.AvgPool2d(2)
        )
        self.down1 = DownTrans(32, 64, 1, heads, head_dim)
        self.down2 = DownTrans(64, 128, 1, heads, head_dim)
        self.down3 = DownTrans(128, 256, 1, heads, head_dim)
        self.output_net = nn.Linear(256, num_classes)

    def forward(self, X):
        input = self.conv1(X)
        # print(input.shape)
        d1 = self.down1(input)
        # print(d1.shape)
        d2 = self.down2(d1)
        # print(d2.shape)
        d3 = self.down3(d2)
        # print(d3.shape)
        d4 = d3.reshape((d3.size(0), d3.size(1), -1))
        # print("d4:", d4.shape)
        d4 = torch.mean(d4, dim=2)
        out = self.output_net(d4)
        out = torch.clip_(out, -1., 1.)
        # print(out.shape)
        return out




