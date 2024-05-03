import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import PIL.Image as Image
import SimpleITK as sitk
import imageio
import numpy as np
import random


def save_model(model, path):
    torch.save(model, path)


def load_model(path, cuda_device):
    if torch.cuda.is_available():
        model = torch.load(path, map_location=torch.device(cuda_device))
    else:
        model = torch.load(path)

    return model


def save_checkpoint(model, optimizer, epoch, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, optimizer, epoch, loss


def read_img(file_path):
    trans = transforms.Compose([transforms.ToTensor()])
    img = Image.open(file_path).convert('L')
    img = trans(img)
    img = img.squeeze()
    return img


def read_mha(file_path):
    sitkimage = sitk.ReadImage(file_path)
    volume = sitk.GetArrayFromImage(sitkimage)
    volume = np.array(volume, dtype=float)
    volume = torch.Tensor(volume).squeeze()
    
    return volume


def save_img(path, img):
    img = img.squeeze()
    img = img.data.cpu().numpy()
    imageio.imwrite(path, img)


def save_mha(path, mha, isVector=False):
    outputs = mha.squeeze()
    outputs = outputs.data.cpu().numpy()
    volume = sitk.GetImageFromArray(outputs, isVector=isVector)
    sitk.WriteImage(volume, path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True