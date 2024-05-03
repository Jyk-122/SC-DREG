import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import numpy as np

from utils import read_mha, save_mha, save_img, read_img


def conv_hu_to_materials_thresholding(hu_values):
    air_mask = (hu_values <= 0)
    soft_mask = (0 < hu_values) * (hu_values <= 0.55)
    bone_mask = (0.55 < hu_values) * (hu_values <= 1)
    air_mask = air_mask.float()
    soft_mask = soft_mask.float()
    bone_mask = bone_mask.float()
    return air_mask, soft_mask, bone_mask


class GenerateDRR(nn.Module):
    def __init__(self):
        super(GenerateDRR, self).__init__()
        tgrid = torch.ones([1, 128, 128, 128, 3])
        for i in range(128):
            scale = np.linspace(101/128, 117/128, 128)[i]
            theta = torch.Tensor([[scale, 0, 0], [0, scale, 0]]).float()
            grid = F.affine_grid(theta.unsqueeze(0), torch.Size([1, 1, 128, 128]))
            tgrid[0, i, :, :, 1:3] = grid
            tgrid[0, i, :, :, 0] = (i - 63.5) / 63.5

        self.tgrid = tgrid.float()

    def forward(self, out_volume):
        device = out_volume.device
        assert out_volume.shape[0] == 1

        tgrid = self.tgrid.to(device)
        output = F.grid_sample(out_volume, tgrid)
        output = output.squeeze(1)
        air, soft, bone = conv_hu_to_materials_thresholding(output)
        air_t = air * output
        soft_t = soft * output
        bone_t = bone * output
        air_t = air_t.sum(1).view(-1, 128, 128).float()
        soft_t = soft_t.sum(1).view(-1, 128, 128).float()
        bone_t = bone_t.sum(1).view(-1, 128, 128).float()
        bone_min = torch.min(torch.min(bone_t, 2, keepdim=True)[0], 1, keepdim=True)[0]
        bone_max = torch.max(torch.max(bone_t, 2, keepdim=True)[0], 1, keepdim=True)[0]
        soft_min = torch.min(torch.min(soft_t, 2, keepdim=True)[0], 1, keepdim=True)[0]
        soft_max = torch.max(torch.max(soft_t, 2, keepdim=True)[0], 1, keepdim=True)[0]
        bone_t = (bone_t - bone_min) / (bone_max - bone_min)
        soft_t = (soft_t - soft_min) / (soft_max - soft_min)
        output_img = bone_t * 0.8 + 0.01 * air_t + 0.5 * soft_t
        output_img = torch.rot90(output_img, 2, (1, 2))
        max_img = torch.max(torch.max(output_img, 2, keepdim=True)[0], 1, keepdim=True)[0]
        min_img = torch.min(torch.min(output_img, 2, keepdim=True)[0], 1, keepdim=True)[0]
        limit = max_img - min_img
        output_img = (output_img - min_img) / limit
        output_img = output_img.view([-1, 1, 128, 128])
        return output_img


class GenerateSeg(nn.Module):
    def __init__(self):
        super(GenerateSeg, self).__init__()
        tgrid = torch.ones([1, 128, 128, 128, 3])
        for i in range(128):
            scale = np.linspace(101/128, 117/128, 128)[i]
            theta = torch.Tensor([[scale, 0, 0], [0, scale, 0]]).float()
            grid = F.affine_grid(theta.unsqueeze(0), torch.Size([1, 1, 128, 128]))
            tgrid[0, i, :, :, 1:3] = grid
            tgrid[0, i, :, :, 0] = (i - 63.5) / 63.5

        self.tgrid = tgrid.float()

    def forward(self, out_volume):
        device = out_volume.device
        assert out_volume.shape[0] == 1

        tgrid = self.tgrid.to(device)
        output = F.grid_sample(out_volume, tgrid, mode='bilinear', align_corners=True)
        output = output.squeeze(1)
        output_img = torch.sum(output, dim=1)
        output_img = torch.tanh(output_img * 100)
        output_img = torch.rot90(output_img, 2, (1, 2))
        return output_img.unsqueeze(0)
        