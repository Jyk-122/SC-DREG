import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import numpy as np
import os

from unet import UNet
from utils import read_mha
from drr import GenerateDRR, GenerateSeg
from unet import UNet
from geometry import perspective, query


class Reg23D(nn.Module):
    def __init__(self, pca_dim):
        super().__init__()

        net = models.resnet34(pretrained=True)
        self.conv1 = net.conv1
        self.pool = net.maxpool
        self.layer0 = nn.Sequential(net.conv1, net.bn1, net.relu)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        self.avg = nn.AvgPool2d(4, stride=1)
        self.fc1 = nn.Linear(512, pca_dim)

    def forward(self, x):
        feat_pyramid = [x]
        x = x.repeat(1, 3, 1, 1)
        x = self.layer0(x)
        feat_pyramid.append(x)
        x = self.layer1(self.pool(x))
        feat_pyramid.append(x)
        x = self.layer2(x)
        feat_pyramid.append(x)
        x = self.layer3(x)
        feat_pyramid.append(x)
        x = self.layer4(x)
        feat_pyramid.append(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        para = self.fc1(x)
        return feat_pyramid, para


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class RefineNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_hidden=32):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_hidden = num_hidden
        self.skip_dim = [1, 64, 64, 128, 256, 512]

        self.skip1 = nn.Sequential(nn.Conv3d(self.skip_dim[0], in_channels, 1, 1, 0),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv3d(in_channels, in_channels, 1, 1, 0),
                                   nn.LeakyReLU(0.2),)
        self.skip2 = nn.Sequential(nn.Conv3d(self.skip_dim[1], num_hidden * 1, 1, 1, 0),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv3d(num_hidden * 1, num_hidden * 1, 1, 1, 0),
                                   nn.LeakyReLU(0.2),)
        self.skip3 = nn.Sequential(nn.Conv3d(self.skip_dim[2], num_hidden * 2, 1, 1, 0),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv3d(num_hidden * 2, num_hidden * 2, 1, 1, 0),
                                   nn.LeakyReLU(0.2),)
        self.skip4 = nn.Sequential(nn.Conv3d(self.skip_dim[3], num_hidden * 4, 1, 1, 0),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv3d(num_hidden * 4, num_hidden * 4, 1, 1, 0),
                                   nn.LeakyReLU(0.2),)
        self.skip5 = nn.Sequential(nn.Conv3d(self.skip_dim[4], num_hidden * 8, 1, 1, 0),
                                   nn.LeakyReLU(0.2),
                                   nn.Conv3d(num_hidden * 8, num_hidden * 8, 1, 1, 0),
                                   nn.LeakyReLU(0.2),)
        
        
        self.conv1 = nn.Sequential(nn.Conv3d(in_channels * 2, num_hidden, 4, 2, 1),
                                   nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv3d(num_hidden * 1 * 2, num_hidden * 2, 4, 2, 1),
                                   nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv3d(num_hidden * 2 * 2, num_hidden * 4, 4, 2, 1),
                                   nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv3d(num_hidden * 4 * 2, num_hidden * 8, 4, 2, 1),
                                   nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv3d(num_hidden * 8 * 2, num_hidden * 8, 1, 1, 0))
        
        self.stack = ResidualStack(num_hidden * 8, num_hidden * 8, 3, num_hidden * 2)
        
        self.conv_trans_1 = nn.Sequential(nn.ConvTranspose3d(num_hidden * 8, num_hidden * 8, 1, 1, 0))
        self.conv_trans_2 = nn.Sequential(nn.ConvTranspose3d(num_hidden * 8, num_hidden * 4, 4, 2, 1),
                                          nn.LeakyReLU(0.2))
        self.conv_trans_3 = nn.Sequential(nn.ConvTranspose3d(num_hidden * 4, num_hidden * 2, 4, 2, 1),
                                          nn.LeakyReLU(0.2))
        self.conv_trans_4 = nn.Sequential(nn.ConvTranspose3d(num_hidden * 2, num_hidden * 1, 4, 2, 1),
                                          nn.LeakyReLU(0.2))
        self.conv_trans_5 = nn.Sequential(nn.ConvTranspose3d(num_hidden * 1, out_channels, 4, 2, 1),
                                          nn.LeakyReLU(0.2))
        
    def forward(self, x, cond):
        c1 = self.skip1(cond[0])
        c2 = self.skip2(cond[1])
        c3 = self.skip3(cond[2])
        c4 = self.skip4(cond[3])
        c5 = self.skip5(cond[4])
        
        x = self.conv1(torch.cat([x, c1], dim=1))
        x = self.conv2(torch.cat([x, c2], dim=1))
        x = self.conv3(torch.cat([x, c3], dim=1))
        x = self.conv4(torch.cat([x, c4], dim=1))
        x = self.conv5(torch.cat([x, c5], dim=1))
        x = self.stack(x)
        x = self.conv_trans_1(x)
        x = self.conv_trans_2(x)
        x = self.conv_trans_3(x)
        x = self.conv_trans_4(x)
        x = self.conv_trans_5(x)
        
        return x


class c2f_model(nn.Module):
    def __init__(self, pca_dim, param_path):
        super().__init__()
        self.pca_dim = pca_dim
        self.param_path = param_path
        
        self.reg23d = Reg23D(pca_dim=pca_dim)
        self.refinenet = RefineNet()
        self.unet = UNet(1, 2, 4)
        
        self.COEFF = torch.Tensor(np.load(os.path.join(param_path, 'coeff4.npy'))).cuda()
        self.mean_ = torch.Tensor(np.load(os.path.join(param_path, 'mean4.npy'))).cuda()
        self.ref = read_mha(os.path.join(param_path, 'ref.nii.gz')).cuda().reshape(1, 1, 128, 128, 128)
        self.ref_seg = read_mha(os.path.join(param_path, 'seg_mandible.nii.gz')).cuda().reshape(1, 1, 128, 128, 128)
        
        self.gen_drr = GenerateDRR()
        self.gen_seg = GenerateSeg()
        
        mesh1, mesh2, mesh3 = np.meshgrid(range(128), range(128), range(128))
        mesh = np.zeros([1, 128, 128, 128, 3])
        mesh[0, :, :, :, 0] = mesh3
        mesh[0, :, :, :, 1] = mesh1
        mesh[0, :, :, :, 2] = mesh2
        self.mesh = torch.from_numpy(mesh).cuda().float()

        self.device = self.mesh.device

    def load_pretrained(self):
        checkpoint = torch.load(os.path.join(self.param_path, 'xray_seg_unet_ckpt.tar'), map_location=self.device)
        self.unet.load_state_dict(checkpoint['model_state_dict'])
        for p in self.unet.parameters():
            p.requires_grad_(False)
    
    def project_feat(self, feat, points):
        calibs = torch.eye(4).unsqueeze(0).cuda()
        out_list = []
        for f in feat:
            out_list.append(query(points, calibs, perspective, f))
            points = F.interpolate(points, scale_factor=0.5, mode='trilinear', align_corners=True)
        return out_list
    
    def forward(self, img):
        logits = self.unet(img)
        
        feat, para = self.reg23d(img)
        coarse_df = torch.mm(para, self.COEFF[:self.pca_dim]) + self.mean_
        coarse_df = coarse_df.reshape(1, 128, 128, 128, 3)
        
        transgrid = coarse_df + self.mesh
        transgrid = (transgrid - 63.5) / 63.5
        coarse_transgrid = transgrid.float()
        
        feat_list = self.project_feat(feat, coarse_transgrid.permute(0, 4, 1, 2, 3))

        coarse_volume = F.grid_sample(self.ref, coarse_transgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
        coarse_vol_seg = F.grid_sample(self.ref_seg, coarse_transgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
        coarse_drr = self.gen_drr(coarse_volume)
        coarse_img_seg = self.gen_seg(coarse_vol_seg)
        
        transgrid = coarse_df.permute(0, 4, 1, 2, 3)
        refine_df = self.refinenet(transgrid, feat_list)
        refine_df = refine_df.permute(0, 2, 3, 4, 1)
        transgrid = refine_df + self.mesh
        transgrid = (transgrid - 63.5) / 63.5
        refine_transgrid = transgrid.float()
        refine_volume = F.grid_sample(self.ref, refine_transgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
        refine_vol_seg = F.grid_sample(self.ref_seg, refine_transgrid, mode='bilinear', padding_mode='zeros', align_corners=True)
        refine_drr = self.gen_drr(refine_volume)
        refine_img_seg = self.gen_seg(refine_vol_seg)
        
        return coarse_drr, refine_drr, coarse_df, refine_df, para, coarse_volume, refine_volume, \
            coarse_img_seg, refine_img_seg, logits, coarse_vol_seg, refine_vol_seg
    