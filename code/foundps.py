import torch
import torch.nn as nn 
import torch.nn.functional as F
from collections import namedtuple
from functools import partial
from tqdm import tqdm

from mimoe import MIMoE
from latentunet import LatentUnet


ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_noise', 'pred_x_start', 'pred_x_T'])

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class FoundPS(nn.Module):
    def __init__(
        self,
        latent_dim = 64,  # hidden_state N dim  == Unet Input Channel 
        num_experts = 16, # num of experts 
        dim = 64, #LatentUNet
        dim_mults=(1, 2, 2, 4), 
        objective = 'pred_x_start', 
        sampling_type = 'pred_x_start',
        timesteps=1000, 
        sampling_timesteps=10,  
        *,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        channels = 1, 
        dim_head = 64,
        image_max_size = 1024, 
        patch_size = 32,  
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16, 
        aux_loss_alpha=0.001,
        condition=True, 
    ):
        super().__init__()

        assert objective in ['pred_noise','pred_x_start']
        assert sampling_type in ['pred_noise','pred_x_start']

        self.mixture_of_experts = MIMoE(
            image_max_size = image_max_size, 
            patch_size = patch_size,  
            dim = latent_dim,  # hidden_state N dim H W
            out_dim = num_experts, # num of experts 
            depth = depth,
            heads = heads,
            mlp_dim = mlp_dim,
            channels = channels, 
            dim_head = dim_head,
            aux_loss_alpha=aux_loss_alpha,
        )

        self.model = LatentUnet(
            dim = dim,  # dim_basement
            dim_mults=dim_mults,
            channels=latent_dim, # Input : xt ms pan -> 2 * channels + 1  # Output : channels
            resnet_block_groups=resnet_block_groups,
            learned_variance=learned_variance,
            learned_sinusoidal_cond=learned_sinusoidal_cond,
            random_fourier_features=random_fourier_features,
            learned_sinusoidal_dim=learned_sinusoidal_dim, 
        )

        self.channels = latent_dim
        self.image_size = image_max_size
        self.condition = condition
        self.objective = objective
        self.sampling_type = sampling_type
        self.num_timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps

        timesteps = 1000
        lamb = 0.0001
        theta_start = 0.0001
        theta_end = 0.02
        thetas = torch.linspace(theta_start, theta_end, timesteps, dtype=torch.float32)
        thetas_cumsum_0_to_t = thetas.cumsum(dim=0)
        thetas_cumsum_0_to_T = thetas_cumsum_0_to_t[-1]
        thetas_cumsum_t_to_T = thetas_cumsum_0_to_T - thetas_cumsum_0_to_t

        sinh_thetas_cumsum_0_to_t = torch.sinh(thetas_cumsum_0_to_t)
        sinh_thetas_cumsum_0_to_T = torch.sinh(thetas_cumsum_0_to_T)
        sinh_thetas_cumsum_t_to_T = torch.sinh(thetas_cumsum_t_to_T)

        Theta = sinh_thetas_cumsum_t_to_T / (sinh_thetas_cumsum_0_to_T) 
        Sigma2 = 2 * lamb * (sinh_thetas_cumsum_0_to_t) * (sinh_thetas_cumsum_t_to_T) / (sinh_thetas_cumsum_0_to_T) 
        Sigma = torch.sqrt(Sigma2) 

        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('thetas', thetas)
        register_buffer('thetas_cumsum_0_to_t', thetas_cumsum_0_to_t)
        register_buffer('thetas_cumsum_0_to_T', thetas_cumsum_0_to_T)
        register_buffer('thetas_cumsum_t_to_T', thetas_cumsum_t_to_T)
        register_buffer('sinh_thetas_cumsum_0_to_t', sinh_thetas_cumsum_0_to_t)
        register_buffer('sinh_thetas_cumsum_0_to_T', sinh_thetas_cumsum_0_to_T)
        register_buffer('sinh_thetas_cumsum_t_to_T', sinh_thetas_cumsum_t_to_T)
        register_buffer('Theta', Theta)
        register_buffer('Sigma2',Sigma2)
        register_buffer('Sigma', Sigma) 
        

    def predict_x_start_from_noise(self, x_t, t, mu, noise):
        return (
            ((x_t - mu - (extract(self.Sigma, t, x_t.shape) * noise)) / (extract(self.Theta, t, x_t.shape))) + mu
        )

    def predict_noise_from_x_start(self, x_t, t, mu, x_start):
        return (
            (x_t - mu - extract(self.Theta, t, x_t.shape)* (x_start - mu) ) / extract(self.Sigma, t, x_t.shape)
        )

    def predict_mu_from_x_t(self, x_t, t, x_start, noise = None):
        """ E[x_T|x_t,x_0] = (x_t - \Theta_t \hat{x_0}) / (1 - \Theta_t)"""
        return (
            (x_t - extract(self.Theta, t, x_t.shape) * x_start) / (1 - extract(self.Theta, t, x_t.shape))
        )

    def model_predictions(self, x_t, mu,  pan, t, clip_denoised=True): 
        model_output = self.model(x_t, mu, pan, t)

        maybe_clip = partial(torch.clamp, min=-1.,max=1.) if clip_denoised else identity

        if self.objective == "pred_noise":
            noise = model_output
            x_start = self.predict_x_start_from_noise(x_t, t, mu, noise) 
            x_start = maybe_clip(x_start)
            x_T = self.predict_mu_from_x_t(x_t, t, x_start, noise)
        elif self.objective == "pred_x_start":
            x_start = model_output
            x_start = maybe_clip(x_start)
            noise = self.predict_noise_from_x_start(x_t, t, mu, x_start)
            x_T = self.predict_mu_from_x_t(x_t, t, x_start, noise)
        else:
            exit('please speficy the prediction mode')

        return ModelResPrediction(noise, x_start, x_T)

    @torch.no_grad()
    def ddim_sample(self, pan,  latent, ms_pixel = None, inv_A = None,  last=True): 
        mu = latent 
        batch = pan.size(0)
        device = self.thetas.device 
        times = torch.linspace(-1, self.num_timesteps - 1,steps=self.sampling_timesteps + 1)#[:num]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
    
        x_t = latent 

        x_start = None
        
        if not last:
            x_t_list = []

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step',disable = True):
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            preds = self.model_predictions(x_t, mu, pan, time_cond)

            noise = preds.pred_noise 
            x_start = preds.pred_x_start 
            x_T = preds.pred_x_T

            if time_next < 0:
                x_t = x_start
                if not last:
                    x_t_list.append(x_t)
                continue

            Theta_now = self.Theta[time]
            Theta_next = self.Theta[time_next]
            Sigma_now = self.Sigma[time]
            Sigma_next = self.Sigma[time_next]
 
            if self.sampling_type == "pred_noise":
                if time == (self.num_timesteps-1):
                    x_t = mu  
                else:
                    x_t = mu +  (Theta_next / Theta_now) * (x_t - mu) - (((Theta_next / Theta_now) * Sigma_now) - Sigma_next) * noise
            elif self.sampling_type == "pred_x_start":
                if time == (self.num_timesteps-1):
                    x_t = mu + Theta_next * (x_start - mu)   
                else:
                    x_t = mu +  (Sigma_next / Sigma_now) * (x_t - mu) + (Theta_next - (Theta_now * Sigma_next / Sigma_now)) *  (x_start - mu)  
            else:
                exit('Illegal objective')
 
            if not last:
                x_t_list.append(x_t)
    
        if self.condition:
            if not last:
                x_t_list = [mu]+x_t_list 
            else:
                x_t_list = [mu, x_t]
            return unnormalize_to_zero_to_one(x_t_list)
        else:
            if not last:
                x_t_list = x_t_list
            else:
                x_t_list = [x_t]
            return unnormalize_to_zero_to_one(x_t_list)

    def grad_and_value(self, x_prev, x_T_hat, y0, inv_A):  
        y_hat = torch.einsum('nchw,ncb->nbhw', x_T_hat, inv_A)
        difference = y_hat # 归一化的问题，还有就是采样的问题
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        return norm_grad, norm

    def dplps_sample(self, pan,  latent, ms_pixel = None,  inv_A = None,  last=True): 
        mu = latent 
        batch = pan.size(0)
        device = self.thetas.device 
        times = torch.linspace(-1, self.num_timesteps - 1,steps=self.sampling_timesteps + 1)#[:num]
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
    
        x_t = latent 

        x_start = None
        
        if not last:
            x_t_list = []

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step',disable = True):
            x_t = x_t.requires_grad_()

            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            preds = self.model_predictions(x_t, mu, pan, time_cond)

            difference = preds.pred_x_start - latent # 归一化的问题，还有就是采样的问题
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_t)[0]
 
            noise = preds.pred_noise 
            x_start = preds.pred_x_start 
            x_T = preds.pred_x_T

            if time_next < 0:
                x_t = x_start
                if not last:
                    x_t_list.append(x_t)
                continue

            Theta_now = self.Theta[time]
            Theta_next = self.Theta[time_next]
            Sigma_now = self.Sigma[time]
            Sigma_next = self.Sigma[time_next]

            scale_ratio = Sigma_now * ( (Theta_next/Theta_now) * Sigma_now - Sigma_next)
            # norm_grad, norm = self.grad_and_value(x_prev=x_t, x_T_hat=x_T, y0=ms_pixel, inv_A = inv_A)   # 有大问题 要归一化回去才能用

            x_t = x_t.detach_()
            
            if self.sampling_type == "pred_noise":
                if time == (self.num_timesteps-1):
                    x_t = mu  
                else:
                    x_t = mu +  (Theta_next / Theta_now) * (x_t - mu) - (((Theta_next / Theta_now) * Sigma_now) - Sigma_next) * noise
            elif self.sampling_type == "pred_x_start":
                if time == (self.num_timesteps-1):
                    x_t = mu + Theta_next * (x_start - mu)   
                else:
                    x_t = mu +  (Sigma_next / Sigma_now) * (x_t - mu) + (Theta_next - (Theta_now * Sigma_next / Sigma_now)) *  (x_start - mu)  
            else:
                exit('Illegal objective')

            x_t = x_t #+ scale_ratio *  norm_grad # + scale_ratio / norm * norm_grad 这里应该有个步数 消融实验
 
            if not last:
                x_t_list.append(x_t)


        if self.condition:
            if not last:
                x_t_list = [mu]+x_t_list 
            else:
                x_t_list = [mu, x_t]
            return unnormalize_to_zero_to_one(x_t_list)
        else:
            if not last:
                x_t_list = x_t_list
            else:
                x_t_list = [x_t]
            return unnormalize_to_zero_to_one(x_t_list)
            
    def sample(self, pan=None, ms = None, last=True): 

        with torch.no_grad():
            ms_latent,_,matrices,_ = self.pixel2latent(pan, ms)
            matrix_pseudo_inv = torch.pinverse(matrices)

            pan_pixel = 2 * pan - 1 
            ms_latent = 2 * ms_latent - 1 

        sample_fn = self.ddim_sample
        # sample_fn = self.dplps_sample
        
        hq_latent = sample_fn(pan = pan_pixel,latent = ms_latent, ms_pixel = ms, inv_A = matrix_pseudo_inv, last=last)[1]# 0 x_T 1 x0
        with torch.no_grad():
            image_sample = torch.einsum('nchw,ncb->nbhw', hq_latent, matrix_pseudo_inv)
            
        return image_sample

    def q_sample(self, x_start, mu, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            mu + (x_start - mu) * extract(self.Theta, t, x_start.shape) + extract(self.Sigma, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self, loss_type='l1'):
        if loss_type == 'l1':
            return F.l1_loss
        elif loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')

    def p_losses(self, pan, ms, gt, t , noise = None): 
        
        ms_latent,hq_latent,matrices,aux_loss = self.pixel2latent(pan, ms, gt)

        pan_pixel = 2 * pan - 1 
        ms_latent = 2 * ms_latent - 1
        hq_latent = 2 * hq_latent - 1
        
        noise = default(noise, lambda: torch.randn_like(hq_latent)) 
         
        x = self.q_sample(hq_latent, ms_latent, t, noise=noise)
        model_out = self.model(x, ms_latent,pan_pixel,t) 
        latent_loss = self.loss_fn(model_out, hq_latent)

        pixel_out = self.latent2pixel(unnormalize_to_zero_to_one(model_out),matrices)
        pixel_loss = self.loss_fn(pixel_out,gt)
        return pixel_loss,latent_loss,aux_loss

    def forward(self, pan, ms, gt):
        b, c, h, w, device, img_size, = *ms.shape, ms.device, self.image_size
        t = torch.randint(0, int(self.num_timesteps), (b,), device=device).long()
        t = torch.clamp(t, min=0, max=self.num_timesteps-1)

        return self.p_losses(pan, ms, gt, t)

    def pixel2latent(self, pan, ms, gt = None):       
        msup = F.interpolate(ms,size=[pan.size(-2),pan.size(-1)])   
        selected_experts,aux_loss = self.mixture_of_experts(pan,msup)
        identity = torch.eye(selected_experts.shape[1], device=selected_experts.device).unsqueeze(0).expand(selected_experts.shape[0], -1, -1)
        matrices = torch.cat([identity, selected_experts[:, :, selected_experts.shape[1]:].softmax(dim=1)], dim=2) 
        ms_latent = torch.einsum('nbhw,nbc->nchw', msup, matrices)
        if gt is not None:
            assert gt.shape == msup.shape
            hq_latent = torch.einsum('nbhw,nbc->nchw', gt, matrices)
        else:
            hq_latent = None
        return ms_latent,hq_latent,matrices,aux_loss
        
    
    def latent2pixel(self,ms_latent_hq,matrices):
        matrices_pseudo_inv = torch.pinverse(matrices)
        image_ps = torch.einsum('nchw,ncb->nbhw', ms_latent_hq, matrices_pseudo_inv)
        return image_ps