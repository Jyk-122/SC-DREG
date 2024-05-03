import os
from torch.utils.data import Dataset
from utils import read_img, read_mha


class Xray_Dataset(Dataset):
    def __init__(self, img_path):
        super().__init__()
        self.img_path = img_path
        self.img_list = sorted(os.listdir(img_path))
    
    def __getitem__(self, index):
        img_name = os.path.join(self.img_path, self.img_list[index])
        img = read_img(img_name).unsqueeze(0)
        return img.cuda()
    
    def __len__(self):
        return len(self.img_list)