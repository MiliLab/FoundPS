import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers.activations import ACT2FN
from router import Router 

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class BasicMLP(nn.Module):
    def __init__(self, hidden_size=512, intermediate_size=1024):
        super(BasicMLP,self).__init__() 
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN["silu"]
        self.act_fn2 = nn.ReLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.act_fn2(down_proj)


class MIMoE(nn.Module):
    def __init__(self,  
                image_max_size = 1024, 
                patch_size = 32,  
                dim = 1024,  # hidden_state N dim
                out_dim = 16, # num of experts 
                depth = 6,
                heads = 16,
                mlp_dim = 2048,
                channels = 1, 
                dim_head = 64,
                aux_loss_alpha=0.001,
            ):
        super().__init__() 
        
        self.hidden_size = dim
        self.num_experts = out_dim

        self.router =  Router(
            image_max_size, 
            patch_size,  
            dim, 
            out_dim,
            depth,
            heads,
            mlp_dim,
            channels, 
            dim_head,
        )
         
        self.experts = nn.ModuleList([(
            BasicMLP(self.hidden_size, self.hidden_size)
        ) for i in range(self.num_experts)])

        self.alpha = aux_loss_alpha

    def forward(self, pan, ms): 

        hidden_states,scores = self.router(torch.cat([pan,ms],dim=1))   
        topk_weight, topk_idx = torch.topk(
                scores, k=ms.size(1), dim=-1, sorted=False
            )  

        bsz, h = hidden_states.shape
        scores_for_aux = scores
        aux_topk = ms.size(1)
        # always compute aux loss based on the naive greedy topk method
        topk_idx_for_aux_loss = topk_idx.view(bsz, -1)
        mask_ce = F.one_hot(
            topk_idx_for_aux_loss.view(-1), num_classes=self.num_experts
        )
        ce = mask_ce.float().mean(0)
        Pi = scores_for_aux.mean(0)
        fi = ce * self.num_experts
        aux_loss = (Pi * fi).sum() * self.alpha

        expert_outputs = torch.stack([expert(hidden_states) for expert in self.experts], dim=1) 
        idx_expanded = topk_idx.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1))  # (B, K, D) 
        selected_experts = torch.gather(expert_outputs, dim=1, index=idx_expanded)

        return selected_experts,aux_loss
    


if __name__ == '__main__':
    print('Hello World')

    criterion = nn.MSELoss()
    
    C = 11
    N = 1
    
    device = f'cuda:{0}'

    model = MIMoE(
        image_max_size = 1024, 
        patch_size = 64,  
        dim = 64,  # hidden_state N dim H W
        out_dim = 16, # num of experts 
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        channels = 1, 
        dim_head = 64,
        aux_loss_alpha=0.001,
    ).to(device)

    ms4 = torch.randn([N,C,256,256]).to(device)
    pan = torch.randn([N,1,256,256]).to(device)

    selected_experts,aux_loss = model(pan,ms4)
    print('Training 256 x 256',selected_experts.shape,aux_loss)
    
    ms4 = torch.randn([N,C,1024,1024]).to(device)
    pan = torch.randn([N,1,1024,1024]).to(device)

    with torch.no_grad():
        selected_experts,aux_loss = model(pan,ms4)

    print('Training 1024 x 1024',selected_experts.shape,aux_loss)
    
    print('Finished')