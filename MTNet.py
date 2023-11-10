# -*- coding:utf-8 -*-
"""
作者：张亦严
日期:2022年10月19日
"""
import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn
from einops import repeat

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads

        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        q = self.to_q(x_qkv[:, 0].unsqueeze(1))
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)

        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = dots.softmax(dim=-1)

        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
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
        return self.to_out(out)

class CrossTransformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, dropout):
        super(CrossTransformer, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, CrossAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim=dim, hidden_dim=dim * 3, dropout=dropout)))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x)
            x = ff(x)
        return x


class Cross_ViT(nn.Module):
    def __init__(self, channel, num_patches, dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.patch_to_embedding1 = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1))
        self.patch_to_embedding2 = nn.Sequential(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=1, p2=1),
                                                 nn.Linear(1, dim, bias=False))
        self.pos_embedding1 = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding2 = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        self.cls_token1 = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token2 = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer_encoder = CrossTransformer(dim, depth, heads, dim_head, dropout=dropout)
        self.cross_attn_layers = nn.ModuleList([])
        for _ in range(1):
            self.cross_attn_layers.append(
                PreNorm(dim, CrossAttention(dim, heads=3, dim_head=64, dropout=0.1)))

    def forward(self, x1, x2):
        x1 = self.patch_to_embedding1(x1)
        x2 = self.patch_to_embedding2(x2)

        b, n, _ = x1.shape

        cls_tokens1 = repeat(self.cls_token1, '() n d -> b n d', b=b)  # [b,1,dim] (16, 1, 32)
        cls_tokens2 = repeat(self.cls_token2, '() n d -> b n d', b=b)  # [b,1,dim] (16, 1, 32)

        x1 = torch.cat((cls_tokens1, x1), dim=1)  # [b,n+1,dim]
        x2 = torch.cat((cls_tokens2, x2), dim=1)  # [b,n+1,dim]

        x1 += self.pos_embedding1[:, :(n + 1)]
        x2 += self.pos_embedding2[:, :(n + 1)]

        x1 = self.transformer_encoder(x1)
        x2 = self.transformer_encoder(x2)
        # for cross_attn in self.cross_attn_layers:
        #     x1_class = x1[:, 0]
        #     x1 = x1[:, 1:]
        #     x2_class = x2[:, 0]
        #     x2 = x2[:, 1:]
        #     # Cross Attn
        #     cat1_q = x1_class.unsqueeze(1)
        #     cat1_qkv = torch.cat((cat1_q, x2), dim=1)
        #     cat1_out = cat1_q + cross_attn(cat1_qkv)
        #     x1 = torch.cat((cat1_out, x1), dim=1)
        #     cat2_q = x2_class.unsqueeze(1)
        #     cat2_qkv = torch.cat((cat2_q, x1), dim=1)
        #     cat2_out = cat2_q + cross_attn(cat2_qkv)
        #     x2 = torch.cat((cat2_out, x2), dim=1)

        return x1, x2

class Hsi_spec(nn.Module):
    def __init__(self, channel, class_num, dim, depth, heads, dim_head, dropout):
        super(Hsi_spec, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=30, kernel_size=3, padding=1),
                                   nn.BatchNorm1d(30),
                                   nn.ReLU(inplace=True))
        self.trans_hsi = Cross_ViT(
                                   channel=channel,
                                   num_patches=30,
                                   dim=dim,
                                   depth=depth,
                                   heads=heads,
                                   dim_head=dim_head,
                                   dropout=dropout)
        self.linear = nn.Linear(dim, class_num)


    def forward(self, x): #(16, 1, 144)
        x = self.conv1(x)
        x = self.trans_hsi(x)
        x = self.linear(x)
        return x

x1 = torch.randn(16, 1, 144)

# net = Hsi_spec(channel=144, class_num=15, dim=30, depth=2, heads=8, dim_head=64, dropout=0.1)
# out = net(x)
# print(out.shape)