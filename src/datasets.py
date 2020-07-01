import torch
import torchvision
from PIL import Image
import os

class T91Set(torch.utils.data.Dataset):
    def __init__(self, image_dir, Y_size = 400, res_factor = 2.5, isTest = False):
        
        target_size = Y_size//res_factor
        self.isTest = isTest
        self.image_dir = image_dir
        self.Y_size = Y_size
        self.res_factor = res_factor
        self.target_size = target_size
        self.image_list = os.listdir(image_dir)
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop((self.Y_size, self.Y_size)),
            torchvision.transforms.ToTensor(),
        ])
        self.test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop((self.Y_size / self.res_factor, self.Y_size / self.res_factor)),
            torchvision.transforms.ToTensor(),
        ])
        self.PIL_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
        ])
        
    def __getitem__(self, index):
        
        if self.isTest:
            image_path = self.image_dir + '/' + str(self.image_list[index])
            image = Image.open(image_path)
            image, _, _ = image.convert('YCbCr').split()
            image = self.test_transforms(image)
            
            LR = self.PIL_transform(image)
            LR = LR.resize((self.Y_size, self.Y_size), Image.BICUBIC)
            LR = self.transforms(LR)
            return LR, str(self.image_list[index])
        
        image_path = self.image_dir + '/' + str(self.image_list[index])
        image = Image.open(image_path)
        image, _, _ = image.convert('YCbCr').split()
        image = self.transforms(image)
        
        HR = image.clone()
        _, h, w = HR.shape
        
        LR = self.PIL_transform(image)
        LR = LR.resize((int(h // self.res_factor), int( w // self.res_factor)), Image.BICUBIC)
        LR = LR.resize((h, w), Image.BICUBIC)
        LR = self.transforms(LR)
        
        return torch.tensor(LR.numpy(), dtype = torch.float), torch.tensor(HR.numpy(), dtype = torch.float)
    
    def __len__(self):
        return len(self.image_list)