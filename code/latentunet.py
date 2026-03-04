import math
from functools import partial

import torch
from torch import einsum, nn
import torch.nn.functional as F 

from einops import rearrange, reduce
from einops.layers.torch import Rearrange
 

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class InfiniteInteractionBlock(nn.Module):
    """
    x, y -> conv encoders -> z1, z2
    then infinite-dim interactions:
      K_exp(p) = exp(alpha * <z1(p), z2(p)>)
      K_geo(p) = 1 / (1 - beta * <z1_hat(p), z2_hat(p)> + eps)
    output: concat features (optionally fused by 1x1 conv)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        pan_dim: int = 1,
        feat_ch: int = 64,
        alpha: float = 1.0,
        beta: float = 1.0,
        use_layernorm: bool = False, 
        eps: float = 1e-6,
        clamp_cos: float = 0.999
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.clamp_cos = clamp_cos

        # Simple encoders: you can replace with your backbone blocks
        self.enc_MS = nn.Sequential(
            nn.Conv2d(latent_dim, feat_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
            nn.Tanh(),
        )
        self.enc_PAN = nn.Sequential(
            nn.Conv2d(pan_dim, feat_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
            nn.Tanh(),
        )
 
        self.use_layernorm = use_layernorm
        if use_layernorm: 
            self.ln = nn.LayerNorm(feat_ch)
 
        concat_ch = feat_ch * 4 
        self.fuse = nn.Sequential(
            nn.Conv2d(concat_ch, feat_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1), 
        ) 

    def _layernorm_2d(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, C, H, W] -> apply LN over C at each (H,W)
        B, C, H, W = z.shape
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]
        z_perm = self.ln(z_perm)
        return z_perm.permute(0, 3, 1, 2).contiguous()

    def forward(self,pan, ms) -> torch.Tensor:
        """
        x: [B, Cx, H, W], y: [B, Cy, H, W]
        returns: concatenated feature map (or fused if fuse_out_ch set)
        """
        z1 = self.enc_MS(ms)  # [B, C, H, W]
        z2 = self.enc_PAN(pan)  # [B, C, H, W]

        if self.use_layernorm:
            z1 = self._layernorm_2d(z1)
            z2 = self._layernorm_2d(z2)
 
        z_exp = torch.exp(self.alpha * (z1 * z2))  # [B,1,H,W]

        # 2) cosine similarity (pattern/shape-sensitive), requires L2 normalization
        z1_hat = F.normalize(z1, p=2, dim=1, eps=self.eps)
        z2_hat = F.normalize(z2, p=2, dim=1, eps=self.eps)
        cos = (z1_hat * z2_hat).clamp(-self.clamp_cos, self.clamp_cos) 

        # geometric-series kernel map (infinite-order interaction with beta^c weights)
        denom = (1.0 - self.beta * cos) + self.eps
        z_geo = 1.0 / denom 

        # concatenate
        out = torch.cat([z1, z2, z_exp, z_geo], dim=1)

        
        # optional fusion
        if self.fuse is not None:
            out = self.fuse(out)

        return out



class InfiniteInteractionBlock2(nn.Module):
    """
    x, y -> conv encoders -> z1, z2
    then infinite-dim interactions:
      K_exp(p) = exp(alpha * <z1(p), z2(p)>)
      K_geo(p) = 1 / (1 - beta * <z1_hat(p), z2_hat(p)> + eps)
    output: concat features (optionally fused by 1x1 conv)
    """

    def __init__(
        self,
        latent_dim: int = 64,
        pan_dim: int = 1,
        feat_ch: int = 64,
        alpha: float = 1.0,
        beta: float = 1.0,
        use_layernorm: bool = False, 
        eps: float = 1e-6,
        clamp_cos: float = 0.999
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.clamp_cos = clamp_cos

        # Simple encoders: you can replace with your backbone blocks
        self.enc_MS = nn.Sequential(
            nn.Conv2d(latent_dim, feat_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
            nn.Tanh(),
        )
        self.enc_PAN = nn.Sequential(
            nn.Conv2d(pan_dim, feat_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1),
            nn.Tanh(),
        )
 
        self.use_layernorm = use_layernorm
        if use_layernorm: 
            self.ln = nn.LayerNorm(feat_ch)
 
        concat_ch = feat_ch * 6
        self.fuse = nn.Sequential(
            nn.Conv2d(concat_ch, feat_ch, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1), 
        ) 

    def _layernorm_2d(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, C, H, W] -> apply LN over C at each (H,W)
        B, C, H, W = z.shape
        z_perm = z.permute(0, 2, 3, 1).contiguous()  # [B,H,W,C]
        z_perm = self.ln(z_perm)
        return z_perm.permute(0, 3, 1, 2).contiguous()

    def forward(self,pan, ms) -> torch.Tensor:
        """
        x: [B, Cx, H, W], y: [B, Cy, H, W]
        returns: concatenated feature map (or fused if fuse_out_ch set)
        """
        z1 = self.enc_MS(ms)  # [B, C, H, W]
        z2 = self.enc_PAN(pan)  # [B, C, H, W]

        if self.use_layernorm:
            z1 = self._layernorm_2d(z1)
            z2 = self._layernorm_2d(z2)
 
        z_exp = torch.exp(self.alpha * torch.cat([z1,z2],dim=1))  # [B,1,H,W]

        # 2) cosine similarity (pattern/shape-sensitive), requires L2 normalization
        z1_hat = F.normalize(z1, p=2, dim=1, eps=self.eps)
        z2_hat = F.normalize(z2, p=2, dim=1, eps=self.eps)
        cos = torch.cat([z1_hat,z2_hat],dim=1).clamp(-self.clamp_cos, self.clamp_cos)  

        # geometric-series kernel map (infinite-order interaction with beta^c weights)
        denom = (1.0 - self.beta * cos) + self.eps
        z_geo = 1.0 / denom 

        # concatenate
        out = torch.cat([z1, z2, z_exp, z_geo], dim=1)

        
        # optional fusion
        if self.fuse is not None:
            out = self.fuse(out)

        return out




class LatentUnet(nn.Module): 
    def __init__(
        self,
        dim, 
        dim_mults=(1, 2, 2, 4),
        channels=64,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16, 
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.depth = len(dim_mults)
        input_channels = channels + channels + 1

        init_dim = dim
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                block_klass(1, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                InfiniteInteractionBlock(latent_dim=dim_in,pan_dim=dim_in,feat_ch=dim_in),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1), 
                nn.AvgPool2d(kernel_size=2, stride=2)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                block_klass(1, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                InfiniteInteractionBlock(latent_dim=dim_out,pan_dim=dim_out,feat_ch=dim_out),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))
 
        self.out_dim = channels

        self.final_res_block = block_klass(dim + channels + 1, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def forward(self, x_t, ms_latent, pan_pixel, time): 
        x = self.init_conv(torch.cat((x_t, ms_latent,pan_pixel), dim=1))
        t = self.time_mlp(time)
        h,s = [],[]
        p = pan_pixel

        for block_ms1, block_ms2, attn_ms, block_pan1, block_pan2, attn_pan, infiniteblock, downsample_ms, downsample_pan in self.downs:
            latent_ms = attn_ms(block_ms2(block_ms1(x, t),t))
            latent_pan = attn_pan(block_pan2(block_pan1(p, t),t))
            x = infiniteblock(latent_pan,latent_ms) 
            h.append(x)
            s.append(p)
            x = downsample_ms(x) 
            p = downsample_pan(p)
             
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block_ms1, block_ms2, attn_ms, block_pan1, block_pan2, attn_pan, infiniteblock, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            latent_ms = attn_ms(block_ms2(block_ms1(x, t),t)) 
            latent_pan = attn_pan(block_pan2(block_pan1(s.pop(), t),t))
            x = infiniteblock(latent_pan,latent_ms)
            x = upsample(x)

        x = torch.cat((x, ms_latent,pan_pixel), dim=1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x) 
        return x 

if __name__ == '__main__':
    print('Hello World')
        
    device = f'cuda:{0}' 

    model = LatentUnet(dim=64,dim_mults=(1, 2, 2, 4),channels=64)
    model.to(device)

    latent_x = torch.rand([2,64,256,256]).to(device)
    pixel_PAN = torch.rand([2,1,256,256]).to(device) 
    time_step = torch.randint(10,(2,)).to(device) 
   
    y = model(x_t = latent_x,ms_latent = latent_x, pan_pixel = pixel_PAN, time = time_step )
    
    print(y.shape,y.max(),y.min())
    print('Finished')