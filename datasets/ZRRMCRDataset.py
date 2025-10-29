import os
import torch
import torch.utils.data
from PIL import Image
from datasets.data_augment import PairCompose, PairToTensor, PairRandomHorizontalFilp
import numpy as np
import imageio
import random

def remove_black_level(img, black_level=63, white_level=4*255):
    img = np.maximum(img-black_level, 0) / (white_level-black_level)
    return img

class LLdataset:
    def __init__(self,dir,patch_size,batch_size,train_file,test_file,num_workers=8):
        # self.config = config

        self.data_dir = dir
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_file = train_file
        self.test_file = test_file

    def get_loaders(self):
        train_dataset = AllWeatherDataset(self.data_dir,
                                          patch_size=self.patch_size,
                                          filelist=self.train_file)
        val_dataset = AllWeatherDataset(self.data_dir,
                                        patch_size=self.patch_size,
                                        filelist=self.test_file, train=False)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size,
                                                   shuffle=True, num_workers=self.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                                 shuffle=False, num_workers=self.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class AllWeatherDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, filelist=None, train=True):
        super().__init__()

        self.dir = dir
        self.file_list = filelist

        self.train = train

        self.train_list = os.path.join(dir, self.file_list)
        with open(self.train_list) as f:
            contents = f.readlines()
            input_names = [i.strip() for i in contents]

        self.input_names = input_names
        self.patch_size = patch_size
        if 'ZRR' in self.dir:
            self.black_level, self.white_level, self.rho = 63, 1020, 8
        elif 'MAI' in self.dir:
            self.black_level, self.white_level, self.rho = 255, 4095, 8
        else:
            raise ValueError('Get wrong dataset name:{}'.format(dir))    

    def random_crop(self, input_img, gt_img):
        _, h,w = input_img.shape
        x0 = random.randrange(start=0,stop=h-self.patch_size,step=2)
        y0 = random.randrange(start=0,stop=w-self.patch_size, step=2)

        input_img_crop = input_img[:,x0:x0+self.patch_size, y0:y0+self.patch_size]

        gt_img_crop = gt_img[:,x0:x0+self.patch_size, y0:y0+self.patch_size]
        # gt_img_gray_crop = gt_img_gray[:,x0:x0+self.patch_size, y0:y0+self.patch_size]

        return input_img_crop, gt_img_crop#, gt_img_gray_crop
    
    def pack_raw(self, raw):


        _, H, W = raw.shape
        

        R = raw[:, 0:H:2, 0:W:2]  #[1, H//2, W//2]
        
 
        G1 = raw[:, 0:H:2, 1:W:2]  #[1, H//2, W//2]
        
  
        G2 = raw[:, 1:H:2, 0:W:2]  #[1, H//2, W//2]

        B = raw[:, 1:H:2, 1:W:2]  #[1, H//2, W//2]
        

        rgbg = torch.cat([R, G1, G2, B], dim=0)
        
        return rgbg

    def get_images(self, index):
        # print(index)
        name = self.input_names[index].replace('\n', '')
        input_name = name.split(' ')[0]

        img_id = input_name.split('/')[-1].split('.')[0]
        if 'ZRR' in self.dir and self.train:
            gt_name  =  name.split(' ')[1]
        else:
            gt_name = name.split(' ')[1]
        
        input_raw, gt_img = np.expand_dims(np.asarray(imageio.imread(os.path.join(self.dir, input_name))), axis=-1), \
            np.asarray(imageio.imread(os.path.join(self.dir, gt_name)))
        
        input_raw = np.maximum((input_raw - self.black_level), 0) / (self.white_level - self.black_level)
        input_raw = torch.tensor(input_raw).permute(2, 0, 1).float()  # 转为[1, H, W]
        input_raw = self.pack_raw(input_raw)  # [4, H//2, W//2]

        gt_img = torch.tensor(gt_img / 255.0).permute(2, 0, 1).float()

        return  {
                    'input_raw': input_raw,
                    'gt_rgb': gt_img,
                    'index':img_id,
                }
        


    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
