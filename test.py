import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_msssim import ssim

import os
from tqdm import tqdm
import numpy as np
import argparse
import sys
sys.path.append('./src/')

from model import c2f_model
from datasets import Xray_Dataset
from utils import read_img, read_mha, save_img, save_mha, save_checkpoint, setup_seed


def test(args):
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    model = c2f_model(args.pca_dim, args.param_path).cuda()
    checkpoint = torch.load(args.ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    input_list = os.listdir(args.input_path)
    for f in input_list:
        if not f.endswith('.png'):
            continue

        path = os.path.join(args.input_path, f)

        img = read_img(path).reshape(1, 1, 128, 128).cuda()

        coarse_drr, refine_drr, coarse_df, refine_df, para, coarse_volume, refine_volume, \
            coarse_img_seg, refine_img_seg, logits, coarse_vol_seg, refine_vol_seg = model(img)

        refine_vol_seg = torch.round(refine_vol_seg)
        
        save_img(os.path.join(output_dir, f'{f[:-4]}_raw.png'), img)
        save_img(os.path.join(output_dir, f'{f[:-4]}_coarse_drr.png'), coarse_drr)
        save_img(os.path.join(output_dir, f'{f[:-4]}_refine_drr.png'), refine_drr)
        save_mha(os.path.join(output_dir, f'{f[:-4]}_refine.nii.gz'), refine_volume)
        save_mha(os.path.join(output_dir, f'{f[:-4]}_seg_reg.nii.gz'), refine_vol_seg)


if __name__ == '__main__':
    setup_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca_dim", type=int, default=60)
    parser.add_argument("--ckpt_path", type=str, default='')
    parser.add_argument("--param_path", type=str, default='')
    parser.add_argument("--input_path", type=str, default='')
    parser.add_argument("--output_dir", type=str, default='./tests')

    args = parser.parse_args()

    test(args)