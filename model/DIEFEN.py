
import torch
import torch.nn as nn

import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple
from einops.layers.torch import Rearrange, Reduce


# DWConv
class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x

# mlp
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.1):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# LKA
class AttentionModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 深度卷积
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        # 深度空洞卷积
        self.conv_spatial = nn.Conv2d(
            dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        # 逐点卷积
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        # 注意力操作
        return u * attn

# attention
class SpatialAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        # 1*1
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        # 激活函数
        self.activation = nn.GELU()
        # LKA
        self.spatial_gating_unit = AttentionModule(d_model)
        # 1*1
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        # res
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.1, drop_path=0.1, act_layer=nn.GELU):
        super().__init__()
        # BN
        self.norm1 = nn.BatchNorm2d(dim)
        # attention
        self.attn = SpatialAttention(dim)
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        # BN2
        self.norm2 = nn.BatchNorm2d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        # FFN
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)


    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1)
                               * self.attn(self.norm1(x)))
        x = x + \
            self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, img_size=(307, 241), patch_size=7, stride=4, in_chans=155, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W

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

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64 * 112 * 35, 128) 
        self.fc2 = nn.Linear(128, 2) 

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 112 * 35)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x


def FeedForward(dim, expansion_factor=4, dropout=0.1, dense=nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )


def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor=4, dropout=0.1):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size=1), nn.Linear

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
        nn.Linear((patch_size ** 2) * channels, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )


class ChannelExchange(nn.Module):

    def __init__(self, p=1 / 2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape

        exchange_map = torch.arange(c) % self.p == 0
        exchange_mask = exchange_map.unsqueeze(0).expand((N, -1))

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[~exchange_mask, ...] = x1[~exchange_mask, ...]
        out_x2[~exchange_mask, ...] = x2[~exchange_mask, ...]
        out_x1[exchange_mask, ...] = x2[exchange_mask, ...]
        out_x2[exchange_mask, ...] = x1[exchange_mask, ...]

        return out_x1, out_x2

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert input_dim % num_heads == 0

        self.input_dim = input_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads

        self.wq = nn.Linear(input_dim, input_dim)
        self.wk = nn.Linear(input_dim, input_dim)
        self.wv = nn.Linear(input_dim, input_dim)

        self.fc = nn.Linear(input_dim, input_dim)

        self.dropout = nn.Dropout(p=dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)


    def forward(self, query, key, value, mask=None):
        batch_size,hw,c = query.shape

        Q = self.wq(query).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.wk(key).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.wv(value).view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.nn.functional.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)

        x = x.permute(0, 2, 1, 3).contiguous().view(batch_size, -1, self.input_dim)
        x = self.fc(x)

        return x

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        self.values = nn.Linear(embed_size, embed_size)

    def forward(self, k, q):
        keys = self.keys(k)
        queries = self.queries(q)
        values = self.values(k)

        attention = torch.matmul(queries, keys.transpose(-2, -1)) / self.embed_size ** 0.5
        attention = F.softmax(attention, dim=-1)

        out = torch.matmul(attention, values)
        return out

class SpatialExchange(nn.Module):
    def __init__(self, p=1 / 2):
        super().__init__()
        assert p >= 0 and p <= 1
        self.p = int(1 / p)

    def forward(self, x1, x2):
        N, c, h, w = x1.shape
        exchange_mask = torch.arange(w) % self.p == 0

        out_x1, out_x2 = torch.zeros_like(x1), torch.zeros_like(x2)
        out_x1[..., ~exchange_mask] = x1[..., ~exchange_mask]
        out_x2[..., ~exchange_mask] = x2[..., ~exchange_mask]
        out_x1[..., exchange_mask] = x2[..., exchange_mask]
        out_x2[..., exchange_mask] = x1[..., exchange_mask]

        return out_x1, out_x2

class DIEFEN(nn.Module):
    def __init__(self, img_size=(450,140), in_chans=155, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4], drop_rate=0.1, drop_path_rate=0.1, norm_layer=nn.LayerNorm,
                 depths=[1,1,1,1], num_stages=4, num_classes=2):
        super().__init__(),
        self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]
        cur = 0

        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(
                img_size=img_size if i == 0 else (img_size[0] // (2 ** (i + 1)), img_size[1] // (2 ** (i + 1))),
                patch_size=7 if i == 0 else 3,
                stride=4 if i == 0 else 2,
                in_chans=in_chans if i == 0 else embed_dims[i - 1],
                embed_dim=embed_dims[i])

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

        self.head = nn.Linear(
            embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.fc1 = nn.Sequential(
            nn.Linear(64, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True),
        )

        self.fc2 = nn.Sequential(
            nn.Linear(128, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True),
        )

        self.fc3 = nn.Sequential(
            nn.Linear(256, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512, bias=True),
        )

        self.fc = nn.Sequential(
            nn.Linear(512*12, 256, bias=True),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2, bias=True),
        )
        self.softmax = nn.Softmax(dim=-1)


    def forward_features(self, x):
        B = x.shape[0]
        features = []

        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm1(x)
        features.append(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm2(x)
        features.append(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm3(x)
        features.append(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm4(x)
        features.append(x)

        return features

    def forward(self, image1, image2):
        spatial_exchange = SpatialExchange(p=0.5)
        channel_exchange = ChannelExchange(p=0.5)
        out_x1,out_x2 = spatial_exchange(image1, image2)
        out_x1, out_x2 = channel_exchange(out_x1, out_x2)
        features1 = self.forward_features(out_x1)
        features2 = self.forward_features(out_x2)

        multihead_attention = MultiHeadAttention(input_dim=64, num_heads=8).to(device)
        features1[0] = multihead_attention(abs(features1[0]-features2[0]),abs(features1[0]-features2[0]), features1[0]) + features1[0]
        features2[0] = multihead_attention(abs(features1[0]-features2[0]),abs(features1[0]-features2[0]), features2[0]) + features2[0]

        multihead_attention = MultiHeadAttention(input_dim=128, num_heads=8).to(device)
        features1[1] = multihead_attention(abs(features1[1]-features2[1]), abs(features1[1]-features2[1]), features1[1]) + features1[1]
        features2[1] = multihead_attention(abs(features1[1]-features2[1]),abs(features1[1]-features2[1]), features2[1]) + features2[1]

        multihead_attention = MultiHeadAttention(input_dim=256, num_heads=8).to(device)
        features1[2] = multihead_attention(abs(features1[2]-features2[2]), abs(features1[2]-features2[2]), features1[2]) + features1[2]
        features2[2] = multihead_attention(abs(features1[2]-features2[2]), abs(features1[2]-features2[2]), features2[2]) + features2[2]

        multihead_attention = MultiHeadAttention(input_dim=512, num_heads=8).to(device)
        features1[3] = multihead_attention(abs(features1[3]-features2[3]), abs(features1[3]-features2[3]), features1[3]) + features1[3]
        features2[3] = multihead_attention(abs(features1[3]-features2[3]), abs(features1[3]-features2[3]), features2[3]) + features2[3]

        # 计算特征之间的差异并相加
        a10 = (features1[0] - features2[0]).mean(dim=1, keepdim=True).squeeze()
        a11 = features1[0].mean(dim=1, keepdim=True).squeeze()
        a12 = features2[0].mean(dim=1, keepdim=True).squeeze()
        a11 = self.fc1(a11)
        a10 = self.fc1(a10)
        a12 = self.fc1(a12)

        a20 = (features1[1] - features2[1]).mean(dim=1, keepdim=True).squeeze()
        a21 = features1[1].mean(dim=1, keepdim=True).squeeze()
        a22 = features2[1].mean(dim=1, keepdim=True).squeeze()
        a20 = self.fc2(a20)
        a21 = self.fc2(a21)
        a22 = self.fc2(a22)

        a30 = (features1[2] - features2[2]).squeeze()
        a31 = features1[2].squeeze()
        a32 = features2[2].squeeze()
        a30 = self.fc3(a30)
        a31 = self.fc3(a31)
        a32 = self.fc3(a32)

        a40 = (features1[3] - features2[3]).squeeze()
        a41 = features1[3].squeeze()
        a42 = features2[3].squeeze()

        sum_features = torch.cat((a10,a20,a30,a40,a11,a12,a21,a22,a31,a32,a41,a42), dim=1)

        deep_out = self.fc(sum_features)

        final_out = self.softmax(deep_out)

        return final_out
