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



    # def random_crop(self, input_img, gt_img, gt_img_gray):
    #     _, h,w = input_img.shape
    #     x0 = random.randrange(start=0,stop=h-self.patch_size,step=2)
    #     y0 = random.randrange(start=0,stop=w-self.patch_size, step=2)

    #     input_img_crop = input_img[:,x0:x0+self.patch_size, y0:y0+self.patch_size]

    #     gt_img_crop = gt_img[:,x0:x0+self.patch_size, y0:y0+self.patch_size]
    #     gt_img_gray_crop = gt_img_gray[:,x0:x0+self.patch_size, y0:y0+self.patch_size]

    #     return input_img_crop, gt_img_crop, gt_img_gray_crop
    def random_crop(self, input_img, gt_img):
        _, h,w = input_img.shape
        x0 = random.randrange(start=0,stop=h-self.patch_size,step=2)
        y0 = random.randrange(start=0,stop=w-self.patch_size, step=2)

        input_img_crop = input_img[:,x0:x0+self.patch_size, y0:y0+self.patch_size]

        gt_img_crop = gt_img[:,x0:x0+self.patch_size, y0:y0+self.patch_size]
        # gt_img_gray_crop = gt_img_gray[:,x0:x0+self.patch_size, y0:y0+self.patch_size]

        return input_img_crop, gt_img_crop#, gt_img_gray_crop
    
    def pack_raw(self, raw):
        """
        将Bayer格式的单通道原始图像转换为RGBG四通道格式
        
        参数:
            raw: 输入的Bayer图像，格式为torch.Tensor，形状为[1, H, W]（C×H×W）
                （注意：输入需为处理过黑电平/白电平后的归一化数据）
        
        返回:
            转换后的四通道图像，形状为[4, H//2, W//2]，通道顺序为R, G1, G2, B
                - R: 红色通道（位于Bayer阵列的(0,0)位置）
                - G1: 绿色通道（位于Bayer阵列的(0,1)位置，红行绿列）
                - G2: 绿色通道（位于Bayer阵列的(1,0)位置，蓝行绿列）
                - B: 蓝色通道（位于Bayer阵列的(1,1)位置）
        """
        # 获取输入图像的高度和宽度（H, W）
        _, H, W = raw.shape
        
        # 提取Bayer阵列中的四个通道：
        # 1. 红色通道（R）：取偶数行、偶数列（0,2,4...行和列）
        R = raw[:, 0:H:2, 0:W:2]  # 形状：[1, H//2, W//2]
        
        # 2. 绿色通道1（G1）：取偶数行、奇数列（红行的绿像素）
        G1 = raw[:, 0:H:2, 1:W:2]  # 形状：[1, H//2, W//2]
        
        # 3. 绿色通道2（G2）：取奇数行、偶数列（蓝行的绿像素）
        G2 = raw[:, 1:H:2, 0:W:2]  # 形状：[1, H//2, W//2]
        
        # 4. 蓝色通道（B）：取奇数行、奇数列
        B = raw[:, 1:H:2, 1:W:2]  # 形状：[1, H//2, W//2]
        
        # 拼接四个通道，形成[4, H//2, W//2]的输出
        rgbg = torch.cat([R, G1, G2, B], dim=0)  # 按通道维度拼接
        
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
        
        # input_img = np.expand_dims(np.asarray(np.asarray(imageio.imread(os.path.join(self.dir, input_name)))), axis=0)
        # gt_img = np.expand_dims(np.asarray(np.asarray(imageio.imread(os.path.join(self.dir, gt_name)))), axis=0)
        
        input_raw, gt_img = np.expand_dims(np.asarray(imageio.imread(os.path.join(self.dir, input_name))), axis=-1), \
            np.asarray(imageio.imread(os.path.join(self.dir, gt_name)))
            # np.expand_dims(np.asarray(imageio.imread(os.path.join(self.dir, gt_name), as_gray=True)), axis=-1)
        
        input_raw = np.maximum((input_raw - self.black_level), 0) / (self.white_level - self.black_level)
        input_raw = torch.tensor(input_raw).permute(2, 0, 1).float()  # 转为[1, H, W]
        input_raw = self.pack_raw(input_raw)  # 输出形状：[4, H//2, W//2]

        # input_raw = np.clip(input_raw, 0, 1)
        # input_raw = np.maximum((input_raw-np.min(input_raw)), 0) / (np.max(input_raw) - np.min(input_raw))
        # input_raw = input_raw / np.max(input_raw)
        # input_raw, gt_img, gt_img_gray = torch.tensor(input_raw).permute(2, 0, 1).float(), \
        #     torch.tensor(gt_img / 255.0).permute(2, 0, 1).float(), \
        #     torch.tensor(gt_img_gray / 255.0).permute(2, 0, 1).float()
        # input_raw, gt_img = torch.tensor(input_raw).permute(2, 0, 1).float(), \
        gt_img = torch.tensor(gt_img / 255.0).permute(2, 0, 1).float()
        # if self.train:
        #     # input_raw, gt_img, gt_img_gray = \
        #     #     self.random_crop(input_raw, gt_img, gt_img_gray)
        #     input_raw, gt_img = \
        #         self.random_crop(input_raw, gt_img)
            # return input_raw, gt_img, gt_img_gray

            # unwarped_gt_name = name.
            # unwarped_gt_img = np.asarray(imageio.imread(unwarped_gt_name))
            # unwarped_gt_img = torch.tensor(unwarped_gt_img/255.0).permute(2, 0, 1).float() 
                # return input_raw, gt_img,img_id
        return  {
                    'input_raw': input_raw,
                    'gt_rgb': gt_img,
                    'index':img_id,
                    # 'input_path':1
                }
        


    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)
