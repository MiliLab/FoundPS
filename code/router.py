from functools import partial
from typing import List, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor, nn  


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def always(val):
    return lambda *args: val

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def divisible_by(numer, denom):
    return (numer % denom) == 0

#feedforward
def FeedForward(dim, hidden_dim):
    return nn.Sequential( 
        nn.Linear(dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, dim),
    )

#attention
class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head *  heads
        self.heads = heads

        self.attend = nn.Softmax(dim = -1)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias = False)
        )

    def forward(
        self,
        x,
        context = None,
        mask = None,
        attn_mask = None
    ):
    
        kv_input = default(context, x)

        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))

        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2))

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

        if exists(attn_mask):
            dots = dots.masked_fill(~attn_mask, -torch.finfo(dots.dtype).max)

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

#transformer block
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head),
                FeedForward(dim, mlp_dim)
            ]))
 
    def forward(
        self,
        x,
        mask = None,
        attn_mask = None
    ):
        for attn, ff in self.layers:
            x = attn(x, mask = mask, attn_mask = attn_mask) + x
            x = ff(x) + x

        return x

class Router(nn.Module):
    def __init__(
        self,
        image_max_size = 1024, 
        patch_size = 32,  
        dim = 1024, 
        out_dim = 4, # num of experts  
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        channels = 1, 
        dim_head = 64
    ):
        super().__init__()
        image_height, image_width = pair(image_max_size)

        assert divisible_by(image_height, patch_size) and divisible_by(image_width, patch_size), 'Image dimensions must be divisible by the patch size.'

        patch_height_dim, patch_width_dim = (image_height // patch_size), (image_width // patch_size)
        patch_dim = channels * (patch_size ** 2)

        self.channels = channels
        self.patch_size = patch_size

        self.to_patch_embedding = nn.Sequential( 
            nn.Linear(patch_dim, dim), 
        )

        self.pos_embed_height = nn.Parameter(torch.randn(patch_height_dim, dim))
        self.pos_embed_width = nn.Parameter(torch.randn(patch_width_dim, dim))

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim)

        self.attn_pool_queries = nn.Parameter(torch.randn(dim))

        self.attn_pool = Attention(dim = dim, dim_head = dim_head, heads = heads)

        self.mlp_head = nn.Sequential( 
            nn.Linear(dim, out_dim, bias = False)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        image,
    ):
        p, device = self.patch_size, self.device

        arange = partial(torch.arange, device = device) 
 
        image_dims = image.shape[-2:]
        assert all([divisible_by(dim, p) for dim in image_dims]), f'height and width {image_dims} of images must be divisible by patch size {p}'

        ph, pw = map(lambda dim: dim // p, image_dims)
        pos = torch.stack(torch.meshgrid((
            arange(ph),
            arange(pw)
        ), indexing = 'ij'), dim = -1)
        patches = rearrange(image, 'b c (h p1) (w p2) ->b (h w c) (p1 p2)', p1 = p, p2 = p)
        patch_positions = rearrange(pos, 'h w c -> (h w) c')
        
        x = self.to_patch_embedding(patches)        

        h_indices, w_indices = patch_positions.unbind(dim = -1)
        h_pos = self.pos_embed_height[h_indices].repeat(1,image.size(1),1)
        w_pos = self.pos_embed_width[w_indices].repeat(1,image.size(1),1)

        x = x + h_pos + w_pos

        x = self.transformer(x)

        queries = repeat(self.attn_pool_queries, 'd -> b n d', n = image.shape[1] - 1, b = x.shape[0])
        x = self.attn_pool(queries, context = x, attn_mask = None) + queries
        hidden_states = (F.softmax(x, dim=1) * x).sum(dim=1) 
        logits = self.mlp_head(hidden_states) 
        scores = torch.softmax(logits, dim=1)  

        return hidden_states,scores


if __name__ == '__main__':
    print('Hello World')

    criterion = nn.MSELoss()
    
    C = 4
    N = 2
    
    device = f'cuda:{0}'

    model = Router(
        image_max_size = 1024, 
        patch_size = 32,  
        dim = 1024, 
        out_dim = 16, # num of experts 
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        channels = 1, 
        dim_head = 64
    ).to(device)
    
    ms4 = torch.randn([N,C,1024,1024]).to(device)

    hidden_states,scores = model(ms4)

    print(hidden_states.shape,scores.shape)
    
    print('Finished')