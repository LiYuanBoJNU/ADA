# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import os
import random
import sys
import math
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.utils as tvu
import torch.nn.functional as F
from ddpm.unet_ddpm import Model
import random
from torchvision.utils import save_image


def random_transforms(img_tensor):
    img_transform1 = transforms.Pad(padding=10)
    img_transform2 = transforms.RandomCrop(100)
    img_transform3 = transforms.CenterCrop(100)
    img_transform4 = transforms.RandomRotation(10)
    img_transform5 = transforms.RandomHorizontalFlip()
    img_transform = transforms.RandomChoice([img_transform1, img_transform2,
                                             img_transform3, img_transform4, img_transform5])
    out_tensor = img_transform(img_tensor)
    out_tensor = F.interpolate(out_tensor, size=(img_tensor.shape[2], img_tensor.shape[3]), mode='bilinear', align_corners=False)
    return out_tensor

def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                        num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,) + (1,) * (len(x_shape) - 1))
    return out



class Diffusion(torch.nn.Module):
    def __init__(self, args, config, facemodel, device=None):
        super().__init__()
        self.args = args
        self.config = config
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.facemodel = facemodel
        self.momentum = args.momentum
        self.a = args.a
        self.alpha = args.alpha
        self.Loss_fn_smth = nn.SmoothL1Loss().to(self.device)

        print("Loading model")
        # if self.config.data.dataset == "CelebA_HQ":
        #     url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        # else:
        #     raise ValueError

        model = Model(self.config)
        # ckpt = torch.hub.load_state_dict_from_url(url, map_location='cpu')
        # model.load_state_dict(ckpt)
        model.load_state_dict(torch.load('celeba_hq.ckpt'))
        model.eval().to(self.device)

        self.model = model

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float()
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
            (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        self.posterior_variance = posterior_variance

        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))
        # print(posterior_variance)
        # # print(len(posterior_variance))
        # print(self.logvar)
        # sys.exit()


    def forward(self, img=None):
        assert isinstance(img, torch.Tensor)
        batch_size = img.shape[0]
        assert img.ndim == 4, img.ndim

        img = F.interpolate(img, size=(256, 256), mode='bilinear', align_corners=False)
        # x_min = clip_by_tensor(img - self.args.max_epsilon * 2 / 255.0, -1.0, 1.0)
        # x_max = clip_by_tensor(img + self.args.max_epsilon * 2 / 255.0, -1.0, 1.0)

        x0 = img.clone()
        grad = torch.zeros_like(x0)

        out_dir = 'test'

        os.makedirs(out_dir, exist_ok=True)
        tvu.save_image((x0 + 1) * 0.5, os.path.join(out_dir, f'original_input.png'))

        e = torch.randn_like(x0)
        total_noise_levels = self.args.t
        a = (1 - self.betas).cumprod(dim=0).to(self.device)
        x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()

        tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'init.png'))

        betas = self.betas.to(self.device)
        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)

        for i in reversed(range(total_noise_levels)):

            t = torch.tensor([i] * batch_size, device=img.device)

            # x_ddpm = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
            with torch.no_grad():
                model_output = self.model(x, t)
                # tvu.save_image((model_output + 1) * 0.5, os.path.join(out_dir, f'model_output_{i}.png'))
            # model_output = F.interpolate(model_output, size=(112, 112), mode='bilinear', align_corners=False)

            weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
            mean = extract(1 / torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape) * model_output)
            logvar = extract(self.logvar, t, x.shape)


            x = x.detach().requires_grad_(True)

            x0_112 = F.interpolate(x0, size=(112, 112), mode='bilinear', align_corners=False)
            x_112 = F.interpolate(x, size=(112, 112), mode='bilinear', align_corners=False)

            loss = - torch.cosine_similarity(self.facemodel(x0_112), self.facemodel(random_transforms(x_112))).mean() #完整实验的损失

            self.facemodel.zero_grad()
            loss.backward()
            noise = x.grad.data
            # tvu.save_image((noise * 1000 + 1) * 0.5, os.path.join(out_dir, f'g_{i}.png'))

            # 计算均值和标准差
            mean_noise = torch.mean(noise)
            std_noise = torch.std(noise)
            # 均值-方差归一化
            noise = (noise - mean_noise) / std_noise

            n = x0 - x

            if i % 10 == 9:
                print(i, loss)
            print(i, loss)
            model_variance = torch.exp(logvar)
            mean = mean + model_variance * noise * self.a + torch.exp(0.5 * logvar) * n

            mask = 1 - (t == 0).float()
            mask = mask.reshape((x.shape[0],) + (1,) * (len(x.shape) - 1))

            x = mean + mask * torch.exp(0.5 * logvar) * torch.randn_like(x)

            x = x.clamp(-1.0, 1.0)

            # added intermediate step vis
            tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'x_t_{i}.png'))

        # x0 = x
        # xs.append(x)
        # torch.save(x0, os.path.join(out_dir, f'samples_{it}.pth'))
        tvu.save_image((x + 1) * 0.5, os.path.join(out_dir, f'samples.png'))

        return x.detach()

