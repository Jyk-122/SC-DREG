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
from datasets import Xray_Dataset, Rec_Dataset
from utils import read_img, read_mha, save_img, save_mha, save_checkpoint, setup_seed


def ncc(p1, p2):
    p1 = p1.data.cpu().numpy()
    p2 = p2.data.cpu().numpy()
    return 1 / (p1.size - 1) * np.sum((p1 - np.mean(p1)) * (p2 - np.mean(p2))) / np.std(p1) / np.std(p2)

def psnr(p1, p2):
    return (10 * torch.log10(1 / F.mse_loss(p1, p2))).item()

def SSIM(p1, p2):
    # p1 = p1.squeeze().unsqueeze(1)
    # p2 = p2.squeeze().unsqueeze(1)
    return ssim(p1, p2, data_range=1, size_average=True).item()


class NCCLoss(nn.Module):
    def __init__(self, win=None, eps=1e-5):
        super(NCCLoss, self).__init__()
        self.win = win
        self.eps = eps

    def ncc(self, I, J):
        ndims = len(list(I.shape)) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions: %d" % ndims

        win = [self.win] * ndims
        # compute CC squares
        I2 = torch.mul(I, I)
        J2 = torch.mul(J, J)
        IJ = torch.mul(I, J)

        # compute filters   *****
        sum_filt = torch.ones([1, 1, self.win, self.win]).to(I.device)
        # strides = 1
        # padding = 4

        # compute local sums via convolution
        I_sum = F.conv2d(input=I, weight=sum_filt, stride=(1, 1), padding=self.win // 2)
        J_sum = F.conv2d(input=J, weight=sum_filt, stride=(1, 1), padding=self.win // 2)
        I2_sum = F.conv2d(input=I2, weight=sum_filt, stride=(1, 1), padding=self.win // 2)
        J2_sum = F.conv2d(input=J2, weight=sum_filt, stride=(1, 1), padding=self.win // 2)
        IJ_sum = F.conv2d(input=IJ, weight=sum_filt, stride=(1, 1), padding=self.win // 2)

        # compute cross correlation
        win_size = self.win * self.win
        # win_size = torch.from_numpy(win_size).to(device)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + self.eps)

        # return negative cc.
        return torch.mean(cc)

    def forward(self, I, J):
        loss = 1 - self.ncc(I, J)
        return loss


class ParaRegularloss(nn.Module):
    def __init__(self):
        super(ParaRegularloss, self).__init__()

    def forward(self, para):
        loss = F.mse_loss(para, torch.zeros_like(para))
        return loss


class CO_DFloss(nn.Module):
    def __init__(self):
        super(CO_DFloss, self).__init__()

    def forward(self, coarse_df, refine_df, coarse_volume=None):
        loss = (coarse_df.clone().detach() - refine_df) ** 2
        loss = torch.mean(loss)
        return loss


class DFRegularloss(nn.Module):
    def __init__(self):
        super(DFRegularloss, self).__init__()

    def forward(self, refine_df, coarse_volume=None):
        c = [1, 1, 1, 1, 1, 1]
        
        up = refine_df[:, :126, 1:127, 1:127, :]
        down = refine_df[:, 2:128, 1:127, 1:127, :]
        front = refine_df[:, 1:127, :126, 1:127, :]
        back = refine_df[:, 1:127, 2:128, 1:127, :]
        left = refine_df[:,  1:127, 1:127, :126, :]
        right = refine_df[:, 1:127, 1:127, 2:128, :]

        mid = torch.zeros_like(refine_df)
        mean = torch.zeros_like(refine_df)
        mid[:, 1:127, 1:127, 1:127, :] = refine_df[:, 1:127, 1:127, 1:127, :]
        mean[:, 1:127, 1:127, 1:127, :] = (c[0] * up + c[1] * down + c[2] * front + c[3] * back + c[4] * left + c[5] * right) / (c[0] + c[1] + c[2] + c[3] + c[4] + c[5])
        
        loss = (mid - mean) ** 2
        loss = torch.mean(loss)
        return loss


class Symmetricloss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, vol):
        return nn.L1Loss()(vol[:, :, :, :, :64], torch.flip(vol[:, :, :, :, 64:], dims=[-1]))


def train(args):
    output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)

    batch_size = args.batch_size
    full_dataset = Xray_Dataset(img_path=args.dataset_path)
    train_loader = DataLoader(dataset=full_dataset, batch_size=batch_size, shuffle=True)

    model = c2f_model(args.pca_dim, args.param_path).cuda()
    model.load_pretrained()

    optimizer = optim.Adam([{'params':model.reg23d.parameters(), 'lr': args.lr_1},
                            {'params':model.refinenet.parameters(), 'lr': args.lr_2}], eps=1e-10)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs * 0.4), int(args.epochs * 0.8)], gamma=0.5, last_epoch=-1)
    
    criterion_para_regular = ParaRegularloss()
    criterion_img = NCCLoss(win=21)
    criterion_df_regular = DFRegularloss()
    criterion_co_df = CO_DFloss()
    criterion_seg = nn.L1Loss()
    criterion_sym = Symmetricloss()

    gamma_s = args.gamma_s
    beta_1 = args.beta_1
    beta_2 = args.beta_2
    gamma_1 = args.gamma_1
    gamma_2 = args.gamma_2
    
    max_ssim = 0
    
    start_epoch = 0

    print("Start training...")

    for epoch in range(start_epoch, args.epochs):
        model.train()

        scheduler.step(epoch=epoch)
        
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            
            img = data
            coarse_drr, refine_drr, coarse_df, refine_df, para, coarse_volume, refine_volume, \
                coarse_img_seg, refine_img_seg, logits, coarse_vol_seg, refine_vol_seg = model(img)
            seg = torch.max(logits, dim=1, keepdim=True)[1].to(torch.float)

            coarse_img_loss = criterion_img(coarse_drr, img)
            refine_img_loss = criterion_img(refine_drr, img)
            para_regular_loss = gamma_2 * gamma_s * criterion_para_regular(para)
            co_df_loss = gamma_2 * beta_1 * criterion_co_df(coarse_df, refine_df, coarse_volume=coarse_volume)
            df_regular_loss = gamma_2 * criterion_df_regular(refine_df, coarse_volume=None)
            seg_loss = gamma_1 * (criterion_seg(seg, refine_img_seg) + criterion_seg(seg, coarse_img_seg))
            seg_loss.requires_grad_(True)
            sym_loss = gamma_2 * beta_2 * (criterion_sym(refine_volume) + criterion_sym(coarse_volume))
            
            loss = coarse_img_loss + refine_img_loss + para_regular_loss + co_df_loss + df_regular_loss + sym_loss + seg_loss
            
            loss.backward()
            optimizer.step()

        save_checkpoint(model, optimizer, epoch, os.path.join(output_dir, 'checkpoint.tar'))
    

if __name__ == '__main__':
    setup_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='c2f_model')
    parser.add_argument("--output_dir", type=str, default='./outputs')
    parser.add_argument("--dataset_path", type=str, default='')
    
    # model
    parser.add_argument("--pca_dim", type=int, default=60)
    parser.add_argument("--param_path", type=str, default='')
    parser.add_argument("--gamma_s", type=float, default=1e-3)
    parser.add_argument("--beta_1", type=float, default=1e-3)
    parser.add_argument("--beta_2", type=float, default=1e-2)
    parser.add_argument("--gamma_1", type=float, default=2)
    parser.add_argument("--gamma_2", type=float, default=2e+2)

    # train
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr_1", type=float, default=1e-4)
    parser.add_argument("--lr_2", type=float, default=1e-4)

    args = parser.parse_args()

    train(args)