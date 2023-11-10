# -*- coding:utf-8 -*-
"""
作者：张亦严
日期:2022年10月13日
"""
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from thop import profile
from einops import repeat, reduce

class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super(CrossAttention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads
        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)
        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)
        q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(Attention, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super(CrossTransformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Vison_transformer(nn.Module):
    def __init__(self, num_patches, dim, depth, heads, dim_head, mlp_dim, dropout=0.0, flag='HSI'):
        super(Vison_transformer, self).__init__()
        # assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        # self.to_patch_embedding = Patch_emd(image_size, patch_size, channels, dim)
        if flag:
            self.patch_to_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),
            )
        else:
            self.patch_to_embedding = nn.Sequential(
                Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),
                nn.Linear(1, dim, bias=False)
            )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
        self.to_latent = nn.Identity()

    def forward(self, x):

        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # [b,1,dim] (16, 1, 32)
        x = torch.cat((cls_tokens, x), dim=1)  # [b,n+1,dim]
        x += self.pos_embedding[:, :(n + 1)]
        x = self.transformer(x)
        x = x[:, 0]
        x = self.to_latent(x)
        return x


class Conv2d_bn_relu(nn.Module):
    def __init__(self, in_channel, out_channel, k, s, p):
        super(Conv2d_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class MTNet(nn.Module):
    def __init__(self, channels, num_patches, dim, depth, heads, dim_head, mlp_dim, num_classes, dropout):  # x1(16, 144, 11, 11) x2(16, 1, 11, 11)
        super(MTNet, self).__init__()
        self.linear = nn.Linear(dim+3*num_patches, num_classes)
        self.linear_f = nn.Linear(num_patches, dim, bias=False)
        self.patch_to_embedding1 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),
        )
        self.patch_to_embedding2 = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (p1 p2 c) (h w)', p1=1, p2=1)
        )
        self.cls_token1 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token3 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token4 = nn.Parameter(torch.randn(1, 1, num_patches*3))

        self.pos_embedding1 = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding2 = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding3 = nn.Parameter(torch.randn(1, 2 * num_patches + 1, dim))
        self.pos_embedding4 = nn.Parameter(torch.randn(1, dim + 1, num_patches * 3))

        self.transformer1 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
        self.transformer2 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
        self.transformer3 = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout)
        self.transformer4 = Transformer(num_patches * 3, depth, heads, dim_head, mlp_dim, dropout=dropout)

        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(1):
            self.cross_attn_layers.append(PreNorm(dim=dim, fn=CrossAttention(dim=dim, heads=4, dim_head=64, dropout=0.1)))
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        self.cat = CrossAttention(dim, heads, dim_head,dropout)

        self.lamda1 = nn.Parameter(torch.tensor([0.1]))
        self.lamda2 = nn.Parameter(torch.tensor([0.1]))
        self.lamda3 = nn.Parameter(torch.tensor([0.1]))
        self.lamda4 = nn.Parameter(torch.tensor([0.1]))
    def forward(self, x1, x2, x_hsi_band): #(16, 144, 7, 7) (16, 1, 7, 7) (16, 147, 144)
       x2_1 = self.conv1(x2) #(16, 144, 7, 7)
       x1_1 = self.patch_to_embedding1(x1) #(16, 49, 144)
       x2_1 = self.patch_to_embedding1(x2_1) #(16, 49, 144)

       b, n, _ = x1_1.shape
       cls_tokens1 = repeat(self.cls_token1, '() n d -> b n d', b=b)
       x1_ = torch.cat([cls_tokens1, x1_1], dim=1)
       x1_ += self.pos_embedding1[:, :(n+1)]
       x1_ = self.transformer1(x1_)

       cls_tokens2 = repeat(self.cls_token2, '() n d -> b n d', b=b)
       x2_ = torch.cat([cls_tokens2, x2_1], dim=1)
       x2_ += self.pos_embedding2[:, :(n+1)]
       x2_ = self.transformer2(x2_)

       x1_other = x1_[:, 1:]
       x1_clstoken = x1_[:, 0].unsqueeze(1)
       x2_other = x2_[:, 1:]
       x2_clstoken = x2_[:, 0].unsqueeze(1)

       x1_new = torch.cat([x2_clstoken, x1_other], dim=1)
       x2_new = torch.cat([x1_clstoken, x2_other], dim=1)

       x1_new_token = self.cat(x1_new).squeeze()
       x2_new_token = self.cat(x2_new).squeeze()

       x_cat = torch.cat([x1_other, x2_other], dim=1) #(16, 242, 144)
       cls_tokens3 = repeat(self.cls_token3, '() n d -> b n d', b=b)
       x3 = torch.cat([cls_tokens3, x_cat], dim=1)
       #x3 += self.pos_embedding3[:, :(n+1)]
       x3 = self.transformer3(x3)

       b1, n1, _ = x_hsi_band.shape
       cls_tokens4 = repeat(self.cls_token4, '() n1 d -> b n1 d', b=b1)
       x4_ = torch.cat([cls_tokens4, x_hsi_band], dim=1)
       x4_ += self.pos_embedding4[:, :(n1 + 1)]
       x4_ = self.transformer4(x4_)

       x1_cls = x1_[:, 0] #HSI
       x2_cls = x2_[:, 0] #LiDAR
       x3_cls = x3[:, 0]  #HSI+LiDAR #(64, 144)
       x4_cls = x4_[:, 0] #(64, 147)
       out = x1_cls + x2_cls + x3_cls

       out = torch.cat([out, x4_cls], dim=-1)
       out = self.linear(out)


       return out

