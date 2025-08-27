from PIL import Image
import cv2
import time
import os
import torch
from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import argparse
import sys
import glob
from torch.utils.data import Dataset
from torchvision.utils import save_image
import tqdm

from ddpm.advdiff_ddpm import Diffusion
# from ddpm.advdiff_ddpm_guided import Diffusion
# from guided_diffusion.advdiff_guided_ddpm import GuidedDiffusion
import yaml
import utils
from utils import str2bool, get_accuracy, get_image_classifier, load_data
import random
# from pytorch_msssim import ssim, ms_ssim
from robFR_model import ArcFace, FaceNet, MobileFace, IR

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM


# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # diffusion models
    parser.add_argument('--config', type=str, default='celeba.yml', help='Path to the config file, celeba, imagenet')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', type=str2bool, default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde]')
    parser.add_argument('--eot_iter', type=int, default=20, help='only for rand version of autoattack')
    parser.add_argument('--use_bm', action='store_true', help='whether to use brownian motion')
    # AdvDiff
    parser.add_argument('--t', type=int, default=100, help='Sampling t step')
    parser.add_argument('--a', type=float, default=15, help='Noise scale')

    # Attack setting
    parser.add_argument('--method', type=str, default='advdiff', help='Choose attack method.')
    parser.add_argument('--model', type=str, default='Mobileface', help='Choose attack model.')
    parser.add_argument('--data_path', type=str, default='data/lfw_diff_1000', help='Input directory with images. data/lfw_diff_1000, assets/datasets/CelebA-HQ')
    parser.add_argument('--save_path', type=str, default='results/test', help='Save images.')
    parser.add_argument("--max_epsilon", type=float, default=12.0, help="Maximum size of adversarial perturbation.")
    parser.add_argument("--num_iter", type=int, default=50, help="Number of iterations.")
    parser.add_argument("--image_resize", type=int, default=112, help="Height of each input images.")
    parser.add_argument("--prob", type=float, default=0.7, help="probability of using diverse inputs.")
    parser.add_argument("--momentum", type=float, default=1.0, help="Momentum")
    parser.add_argument("--alpha", type=float, default=8 / 255, help="alpha")
    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = utils.dict2namespace(config)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    new_config.device = device
    print(args)

    # # set random seed
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


class FaceDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

    def __getitem__(self, idx):
        sample = self.samples[idx]

        img = Image.open(sample)
        img = self.transformer(img)
        return img, 1

    def __len__(self):
        return len(self.samples)


def attack(args, config):
    if args.model == 'Arcface':
        model = ArcFace.IR_50((112, 112))
        model.load_state_dict(torch.load('robFR_model/model_ir_se50.pth'))
        model.eval()
        model.cuda()

    elif args.model == 'Mobileface':
        model = MobileFace.MobileFacenet()
        model.load_state_dict(torch.load('robFR_model/Backbone_mobileface_Epoch_36_Batch_409392_Time_2019-04-07-16-40_checkpoint.pth'))
        model.eval()
        model.cuda()

    elif args.model == 'IR50':
        model = IR.IR_50((112, 112))
        model.load_state_dict(torch.load('robFR_model/IR50_Softmax.pth'))
        model.eval()
        model.cuda()
    else:
        print("error model")
    print('run model:', args.model)

    if args.method == 'advdiff':
        run_attack = Diffusion(args, config, facemodel=model, device=config.device)
        print('t:', args.t)
    # elif args.method == 'advdiff_guided':
    #     run_attack = GuidedDiffusion(args, config, facemodel=model, device=config.device)
    #     print('t:', args.t)
    else:
        print("error method")
    print('run method:', args.method)

    print('data path:', args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    print('save path:', args.save_path)

    paths = glob.glob(args.data_path + '/*.png') + glob.glob(args.data_path + '/*.jpg')
    paths.sort(reverse=False)
    print(len(paths))
    # print(paths)
    # sys.exit()
    dataset = FaceDataset(paths)
    if args.method == 'Admix':
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)
    else:
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    i = 0
    # save_paths_len = glob.glob(args.save_path + '/*.png')
    for images, _ in data_loader:
        print('i:', i, paths[i])
        t0 = time.time()
        images = images.cuda()
        # images = F.interpolate(images, size=[112,112], mode='bilinear').cuda()

        adv_img = run_attack.forward(images)
        # print(adv_img)
        adv_img = F.interpolate(adv_img, size=(112, 112), mode='bilinear', align_corners=False)
        adv_feat = model(adv_img)
        x_feat = model(images)
        cos = torch.cosine_similarity(adv_feat, x_feat)
        print(cos)
        print((adv_img-images).max(), (adv_img-images).min())

        outimg = (adv_img + 1) / 2
        outimg = outimg.clamp(0.0, 1.0)
        inimg = (images + 1) / 2
        inimg = inimg.clamp(0.0, 1.0)
        ssim_val = ssim(outimg, inimg, data_range=1, size_average=False)
        l2_norm = torch.norm(outimg - inimg, p=2)
        print(ssim_val, l2_norm)
        for saveimg in outimg:
            # print(saveimg.shape)
            outpath = os.path.join(args.save_path, os.path.basename(paths[i]))
            # outpath = args.save_path + os.path.basename(paths[i])
            save_image(saveimg, outpath)
            i += 1

        t1 = time.time()
        print('time:', t1 - t0)


if __name__ == '__main__':
    args, config = parse_args_and_config()
    attack(args, config)
