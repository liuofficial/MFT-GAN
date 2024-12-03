import torch
import torch.nn as nn
from einops import rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward_1(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.ffn(x)

class FeedForward_2(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim//2, dim//2),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.ffn(x)

class Spatial_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        # project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, N, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b N (h d) -> b h N d', h=h), qkv)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h N d -> b N (h d)')
        out = self.to_out(out)
        return out

class Spectral_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        # project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, c, _ = x.shape
        h = self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b c (h d) -> b h c d', h=h), qkv)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h c d -> b c (h d)')
        out = self.to_out(out)
        return out

class Spatial_Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Spatial_Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward_1(dim, mlp_dim, dropout=dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Spatial_Transformer_Block(nn.Module):
    def __init__(self, in_c, out_c, depth, heads, dim_head):
        super().__init__()
        self.Embedding = nn.Conv2d(in_c, out_c, (1, 1))
        self.transformer = Spatial_Transformer(dim=out_c, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=out_c//2, dropout=0.)
    def forward(self,X):
        H = X.size(2)
        E = self.Embedding(X)
        E = rearrange(E, 'B c H W -> B (H W) c', H=H)
        out1 = self.transformer(E)
        out = rearrange(out1, 'B (H W) C -> B C H W', H=H)
        return out


class DownTrans(nn.Module):
    def __init__(self, in_c, out_c, depth, heads, dim_head):
        super().__init__()
        self.net = nn.Sequential(
            Spatial_Transformer_Block(in_c, out_c, depth, heads, dim_head),
            nn.AvgPool2d(2),
        )
    def forward(self,X):
        out = self.net(X)
        return out

class Spetral_Transformer_Block(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.dim_list = [dim,dim//2,dim//4]
        self.layers = nn.ModuleList([])
        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(self.dim_list[i], Spectral_Attention(self.dim_list[i], heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(self.dim_list[i], FeedForward_2(self.dim_list[i],dropout=dropout))
            ]))

    def forward(self, x):
        for attn,  ff in self.layers:
            x = attn(x)
            x = ff(x)
            # print(x.shape)
        return x


class Cross_Self_Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout=0.5):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv_1 = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_qkv_2 = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x,y):
        b, N, _ = x.shape
        h = self.heads
        qkv1 = self.to_qkv_1(x).chunk(3, dim=-1)
        q1, k1, v1 = map(lambda t: rearrange(t, 'b N (h d) -> b h N d', h=h), qkv1)

        qkv2 = self.to_qkv_2(y).chunk(3, dim=-1)
        q2, k2, v2 = map(lambda t: rearrange(t, 'b N (h d) -> b h N d', h=h), qkv2)

        dots1 = torch.einsum('b h i d, b h j d -> b h i j', q2, k1) * self.scale
        attn1 = dots1.softmax(dim=-1)
        out1 = torch.einsum('b h i j, b h j d -> b h i d', attn1, v1)
        out1 = rearrange(out1, 'b h N d -> b N (h d)')

        dots2 = torch.einsum('b h i d, b h j d -> b h i j', q1, k2) * self.scale
        attn2 = dots2.softmax(dim=-1)
        out2 = torch.einsum('b h N N, b h N d -> b h N d', attn2, v2)
        out2 = rearrange(out2, 'b h N d -> b N (h d)')

        input1 = torch.add(out1,out2)
        input2 = self.to_out(input1)

        output = input2 + x + y

        return output

class CSABlock(nn.Module):
    def __init__(self, in_c1, in_c2):
        super().__init__()
        self.Embedding1 = nn.Conv2d(in_c1,  in_c1, (1, 1))
        self.Embedding2 = nn.Conv2d(in_c2, in_c1, (1, 1))
        self.atten = Cross_Self_Attention(dim=in_c1, heads=4, dim_head=18)
        self.FFN = FeedForward_1(in_c1, in_c1//2, 0.0)
        
    def forward(self, X, Y):
        H = X.size(2)
        E1 = self.Embedding1(X)
        E11 = rearrange(E1, 'B c H W -> B (H W) c', H=H)

        E2 = self.Embedding2(Y)
        E22 = rearrange(E2, 'B c H W -> B (H W) c', H=H)
        attn = self.atten(E11, E22)
    
        out = self.FFN(attn)
        out = attn + out

        output = rearrange(out, 'B (H W) C -> B C H W', H=H)
    
        return output

