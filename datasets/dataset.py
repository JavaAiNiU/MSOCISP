import torch
from torch.utils.data import Dataset
import os
import numpy as np
import glob
import rawpy
from util import *
from torch.utils.data import DataLoader
class SonyDataset(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps
        # get train IDs
        # glob.glob 函数用于匹配指定模式的文件路径，返回匹配的文件路径列表。
        # gt_dir + '0*.ARW' 构建了一个路径模式，表示在 gt_dir 目录下找到以0开头的所有ARW文件。
        # 这样就获取了所有以0开头的ARW文件的文件路径列表
        self.fns = glob.glob(gt_dir + '0*.ARW') # file names
        # '/raid/hbj/data/Sony/long/00064_00_10s.ARW' # 总长度为161
        # 提取每个文件路径中的文件名部分、取文件名中的前五个字符提取了训练ID、将提取的字符串转换为整数 
        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]
        # [64, 17, 42, 15, 119, 138, 108, 36, 85, 145, 10, 148, 56, 100, ...] 161张
        # 主要用于管理原始数据，跟踪损失以及确定已保存的模型检查点的最新时期
        # 创建了一个长度为6000的列表，用于存储地面实况图像。每个元素初始化为None。
        # Raw data takes long time to load. Keep them in memory after loaded.
        self.gt_images = [None] * 6000
        self.input_images = dict()
        self.input_images['300'] = [None] * len(self.ids)
        self.input_images['250'] = [None] * len(self.ids)
        self.input_images['100'] = [None] * len(self.ids)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, ind):
        # get the path from image id
        id = self.ids[ind] # id为186
        in_files = glob.glob(self.input_dir + '%05d_00*.ARW' % id)
        # '/raid/hbj/data/Sony/short/00186_00_0.1s.ARW'
        # '/raid/hbj/data/Sony/short/00186_00_0.033s.ARW'
        # '/raid/hbj/data/Sony/short/00186_00_0.04s.ARW'
        in_path = in_files[np.random.random_integers(0, len(in_files) - 1)]
        # '/raid/hbj/data/Sony/short/00186_00_0.04s.ARW'
        in_fn = os.path.basename(in_path) # '00186_00_0.04s.ARW'
        
        # 使用 glob 函数获取与当前训练ID相关的地面实况（长曝光）图像的文件路径列表
        gt_files = glob.glob(self.gt_dir + '%05d_00*.ARW' % id) # '/raid/hbj/data/Sony/long/00186_00_10s.ARW'
        gt_path = gt_files[0] # '/raid/hbj/data/Sony/long/00186_00_10s.ARW'
        gt_fn = os.path.basename(gt_path) # '00186_00_10s.ARW'

        in_exposure = float(in_fn[9:-5]) # 0.04
        gt_exposure = float(gt_fn[9:-5]) # 10
        ratio = min(gt_exposure / in_exposure, 300) # 250.0

        if self.input_images[str(ratio)[0:3]][ind] is None:
            raw = rawpy.imread(in_path)
            self.input_images[str(ratio)[0:3]][ind] = np.expand_dims(pack_raw(raw), axis=0) * ratio
            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            self.gt_images[ind] = np.expand_dims(np.float32(im / 65535.0), axis=0)
        # 类型为<class 'numpy.ndarray'>
        # print(f"self.gt_images[ind]的尺寸大小为{self.gt_images[ind]}。self.gt_images[ind]的类型为{type(self.gt_images[ind])}")

        # crop
        H = self.input_images[str(ratio)[0:3]][ind].shape[1] # 1424
        W = self.input_images[str(ratio)[0:3]][ind].shape[2] # 2128
        
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        input_patch = self.input_images[str(ratio)[0:3]][ind][:, yy:yy + self.ps, xx:xx + self.ps, :] # (1, 512, 512, 4)  dtype:'float32'
        gt_raw = 
        gt_patch = self.gt_images[ind][:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]# (1, 1024, 1024, 3)

        # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0) # 1.0-0.0
        # print("*"*50)
        
        # 将NumPy数组input_patch转换为PyTorch张量
        input_patch = torch.from_numpy(input_patch)
        # print(f"input_patch的大小为{input_patch.shape},而input_patch的类型为{type(input_patch)}") # input_patch的大小为torch.Size([1, 512, 512, 4]),而input_patch的类型为<class 'torch.Tensor'>
        input_patch = torch.squeeze(input_patch)
        # print(f"input_patch的大小为{input_patch.shape},而input_patch的类型为{type(input_patch)}") # input_patch的大小为torch.Size([512, 512, 4]),而input_patch的类型为<class 'torch.Tensor'>
        input_patch = input_patch.permute(2, 0, 1)
        # print(f"input_patch的大小为{input_patch.shape},而input_patch的类型为{type(input_patch)}") # input_patch的大小为torch.Size([4, 512, 512]),而input_patch的类型为<class 'torch.Tensor'>
        
        # 将NumPy数组gt_patch转换为PyTorch张量    
        gt_patch = torch.from_numpy(gt_patch)
        # print(f"gt_patch的大小为{gt_patch.shape},而gt_patch的类型为{type(gt_patch)}") # gt_patch的大小为torch.Size([1, 1024, 1024, 3]),而gt_patch的类型为<class 'torch.Tensor'>
        # 用于从张量中移除尺寸为1的维度
        gt_patch = torch.squeeze(gt_patch)
        # print(f"gt_patch的大小为{gt_patch.shape},而gt_patch的类型为{type(gt_patch)}") # gt_patch的大小为torch.Size([1024, 1024, 3]),而gt_patch的类型为<class 'torch.Tensor'>
        gt_patch = gt_patch.permute(2, 0, 1)
        # print(f"gt_patch的大小为{gt_patch.shape},而gt_patch的类型为{type(gt_patch)}") # gt_patch的大小为torch.Size([3, 1024, 1024]),而gt_patch的类型为<class 'torch.Tensor'>
        # print("*"*50)
        
        return input_patch, gt_patch, id, ratio


class SonyTestDataset(Dataset):
    def __init__(self, input_dir, gt_dir):
        self.input_dir = input_dir # '/raid/hbj/data/Sony/short/'
        self.gt_dir = gt_dir       # '/raid/hbj/data/Sony/long/'

        self.fns = glob.glob(input_dir + '1*.ARW')  # file names, 1 for testing.
        # ['/raid/hbj/data/Sony/short/10011_01_0.04s.ARW', '/raid/hbj/data/Sony/short/10228_01_0.1s.ARW', '/raid/hbj/data/Sony/short/10035_10_0.1s.ARW', '/raid/hbj/data/Sony/short/10170_06_0.1s.ARW', '/raid/hbj/data/Sony/short/10191_04_0.04s.ARW', '/raid/hbj/data/Sony/short/10006_08_0.1s.ARW', '/raid/hbj/data/Sony/short/10035_02_0.1s.ARW', '/raid/hbj/data/Sony/short/10139_08_0.1s.ARW', '/raid/hbj/data/Sony/short/10228_06_0.04s.ARW', '/raid/hbj/data/Sony/short/10228_07_0.04s.ARW', '/raid/hbj/data/Sony/short/10163_04_0.1s.ARW', '/raid/hbj/data/Sony/short/10030_02_0.1s.ARW', '/raid/hbj/data/Sony/short/10198_07_0.04s.ARW', '/raid/hbj/data/Sony/short/10139_09_0.1s.ARW', ...]
        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns] # '/raid/hbj/data/Sony/short/10011_01_0.04s.ARW'
        # [10011, 10228, 10035, 10170, 10191, 10006, 10035, 10139, 10228, 10228, 10163, 10030, 10198, 10139, ...]
    def __len__(self):
        return len(self.ids) # 

    def __getitem__(self, ind):
        # input 获取对应索引的图像ID
        # in_path 和 gt_path：根据图像ID构建输入图像和地面真实图像的路径。
        # in_fn 和 gt_fn：提取输入图像和地面真实图像的文件名。
        # in_exposure 和 gt_exposure：从文件名中提取输入图像和地面真实图像的曝光值。
        # ratio：计算曝光值之比，用于后续图像亮度调整。
        id = self.ids[ind]
        in_path = self.fns[ind]
        in_fn = os.path.basename(in_path)
        # ground truth 
        gt_files = glob.glob(self.gt_dir + '%05d_00*.ARW' % id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        # ratio
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        # load images
        # 处理输入RAW图像数据，进行亮度调整，并转换为PyTorch张量
        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
        
        # 对输入图像进行后处理，包括白平衡、无自动亮度调整
        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
        scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
        
        # gt_full：加载地面真实RAW图像数据，进行相同的后处理步骤。
        gt_raw = rawpy.imread(gt_path)
        im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)

        # clipping, convert to tensor
        input_full = np.minimum(input_full, 1.0)  # 确保input_full数组中的所有值都不超过1.0，即将亮度范围限制在[0,1]之间
        input_full = torch.from_numpy(input_full) # 将NumPy数组input_full转换为PyTorch张量
        input_full = torch.squeeze(input_full)    # 移除input_full张量中的所有尺寸为1的维度，使其变为一个三维张量
        input_full = input_full.permute(2, 0, 1)  # 调整张量的通道顺序，将通道维度（通常是最后一个维度）移到最前面，这通常是因为PyTorch默认使用通道-高度-宽度（CHW）的顺序。

        scale_full = torch.from_numpy(scale_full) # 将NumPy数组scale_full转换为PyTorch张量
        scale_full = torch.squeeze(scale_full)    # 移除scale_full张量中的所有尺寸为1的维度

        gt_full = torch.from_numpy(gt_full)       # 将NumPy数组gt_full转换为PyTorch张量
        gt_full = torch.squeeze(gt_full)          # gt_full = torch.squeeze(gt_full): 移除gt_full张量中的所有尺寸为1的维度
        # print(f"input_full的大小为{input_full.shape},类型为{type(input_full)}")
        # print(f"scale_full的大小为{scale_full.shape},类型为{type(scale_full)}")
        # print(f"gt_full的大小为{gt_full.shape},类型为{type(gt_full)}")
        # input_full的大小为torch.Size([4, 1424, 2128]),类型为<class 'torch.Tensor'>
        # scale_full的大小为torch.Size([2848, 4256, 3]),类型为<class 'torch.Tensor'>
        # gt_full的大小为torch.Size([2848, 4256, 3]),类型为<class 'torch.Tensor'>
        return input_full, scale_full, gt_full, id, ratio

class SonyTestCropDataset(Dataset):
    def __init__(self, input_dir, gt_dir):
        self.input_dir = input_dir # '/raid/hbj/data/Sony/short/'
        self.gt_dir = gt_dir       # '/raid/hbj/data/Sony/long/'

        self.fns = glob.glob(input_dir + '1*.ARW')  # file names, 1 for testing.
        # ['/raid/hbj/data/Sony/short/10011_01_0.04s.ARW', '/raid/hbj/data/Sony/short/10228_01_0.1s.ARW', '/raid/hbj/data/Sony/short/10035_10_0.1s.ARW', '/raid/hbj/data/Sony/short/10170_06_0.1s.ARW', '/raid/hbj/data/Sony/short/10191_04_0.04s.ARW', '/raid/hbj/data/Sony/short/10006_08_0.1s.ARW', '/raid/hbj/data/Sony/short/10035_02_0.1s.ARW', '/raid/hbj/data/Sony/short/10139_08_0.1s.ARW', '/raid/hbj/data/Sony/short/10228_06_0.04s.ARW', '/raid/hbj/data/Sony/short/10228_07_0.04s.ARW', '/raid/hbj/data/Sony/short/10163_04_0.1s.ARW', '/raid/hbj/data/Sony/short/10030_02_0.1s.ARW', '/raid/hbj/data/Sony/short/10198_07_0.04s.ARW', '/raid/hbj/data/Sony/short/10139_09_0.1s.ARW', ...]
        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns] # '/raid/hbj/data/Sony/short/10011_01_0.04s.ARW'
        # [10011, 10228, 10035, 10170, 10191, 10006, 10035, 10139, 10228, 10228, 10163, 10030, 10198, 10139, ...]
    def __len__(self):
        return len(self.ids) # 

    def __getitem__(self, ind):
        # input 获取对应索引的图像ID
        # in_path 和 gt_path：根据图像ID构建输入图像和地面真实图像的路径。
        # in_fn 和 gt_fn：提取输入图像和地面真实图像的文件名。
        # in_exposure 和 gt_exposure：从文件名中提取输入图像和地面真实图像的曝光值。
        # ratio：计算曝光值之比，用于后续图像亮度调整。
        id = self.ids[ind]
        in_path = self.fns[ind]
        input_image = os.path.basename(in_path)
        
        # ground truth 
        gt_files = glob.glob(self.gt_dir + '%05d_00*.ARW' % id)
        gt_path = gt_files[0]
        gt_image = os.path.basename(gt_path)
        
        # ratio
        in_exposure = float(input_image[9:-5])
        gt_exposure = float(gt_image[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)

        # load images
        # 处理输入RAW图像数据，进行亮度调整，并转换为PyTorch张量
        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio
        
        # 对输入图像进行后处理，包括白平衡、无自动亮度调整
        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
        scale_image = np.expand_dims(np.float32(im / 65535.0), axis=0)
        
        # gt_full：加载地面真实RAW图像数据，进行相同的后处理步骤。
        gt_raw = rawpy.imread(gt_path)
        gt_image = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_image = np.expand_dims(np.float32(gt_image / 65535.0), axis=0)
        
        H = self.input_images[ind].shape[1]
        W = self.input_images[ind].shape[2]
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        
        input_patch = input_image[:, yy:yy + self.ps, xx:xx + self.ps, :]
        scale_patch = scale_image[:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]
        gt_patch = gt_image[:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]
        
        # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            scale_patch = np.flip(scale_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            scale_patch = np.flip(scale_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            scale_patch = np.transpose(scale_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
        

        # clipping, convert to tensor
        input_patch = np.minimum(input_patch, 1.0)  # 确保input_full数组中的所有值都不超过1.0，即将亮度范围限制在[0,1]之间
        input_patch = torch.from_numpy(input_patch) # 将NumPy数组input_full转换为PyTorch张量
        input_patch = torch.squeeze(input_patch)    # 移除input_full张量中的所有尺寸为1的维度，使其变为一个三维张量
        input_patch = input_patch.permute(2, 0, 1)  # 调整张量的通道顺序，将通道维度（通常是最后一个维度）移到最前面，这通常是因为PyTorch默认使用通道-高度-宽度（CHW）的顺序。

        scale_patch = torch.from_numpy(scale_patch) # 将NumPy数组scale_full转换为PyTorch张量
        scale_patch = torch.squeeze(scale_patch)    # 移除scale_full张量中的所有尺寸为1的维度

        gt_patch = torch.from_numpy(gt_patch)       # 将NumPy数组gt_full转换为PyTorch张量
        gt_patch = torch.squeeze(gt_patch)          # gt_full = torch.squeeze(gt_full): 移除gt_full张量中的所有尺寸为1的维度
        # print(f"input_full的大小为{input_full.shape},类型为{type(input_full)}")
        # print(f"scale_full的大小为{scale_patch.shape},类型为{type(scale_patch)}")
        # print(f"gt_full的大小为{gt_patch.shape},类型为{type(gt_patch)}")
        # input_full的大小为torch.Size([4, 1424, 2128]),类型为<class 'torch.Tensor'>
        # scale_full的大小为torch.Size([2848, 4256, 3]),类型为<class 'torch.Tensor'>
        # gt_full的大小为torch.Size([2848, 4256, 3]),类型为<class 'torch.Tensor'>
        return input_patch, scale_patch, gt_patch, id, ratio



   
class SonyData(Dataset):
    def __init__(self, input_dir, gt_dir, ps=512):
        self.input_dir = input_dir
        self.gt_dir = gt_dir
        self.ps = ps
        
        self.fns = glob.glob(input_dir + '0*.ARW')  # file names, 1 for testing.

        self.ids = [int(os.path.basename(fn)[0:5]) for fn in self.fns]
        
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, ind):
        # input
        id = self.ids[ind]
        in_path = self.fns[ind]
        in_fn = os.path.basename(in_path)
        # ground truth
        gt_files = glob.glob(self.gt_dir + '%05d_00*.ARW' % id)
        gt_path = gt_files[0]
        gt_fn = os.path.basename(gt_path)
        # ratio
        in_exposure = float(in_fn[9:-5])
        gt_exposure = float(gt_fn[9:-5])
        ratio = min(gt_exposure / in_exposure, 300)
        # load images
        raw = rawpy.imread(in_path)
        input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

        im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        # scale_full = np.expand_dims(np.float32(im/65535.0),axis = 0)*ratio
        scale_full = np.expand_dims(np.float32(im / 65535.0), axis=0)
        # crop
        H = input_full.shape[1]
        W = input_full.shape[2]
        
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        
        input_patch = input_full[:, yy:yy + self.ps, xx:xx + self.ps, :]
        gt_patch = scale_full[:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]
        
         # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()
        
        
        input_patch = np.minimum(input_patch, 1.0)
        # print("*"*50)
        
        # 将NumPy数组input_patch转换为PyTorch张量
        input_patch = torch.from_numpy(input_patch)
        # print(f"input_patch的大小为{input_patch.shape},而input_patch的类型为{type(input_patch)}") # input_patch的大小为torch.Size([1, 512, 512, 4]),而input_patch的类型为<class 'torch.Tensor'>
        input_patch = torch.squeeze(input_patch)
        # print(f"input_patch的大小为{input_patch.shape},而input_patch的类型为{type(input_patch)}") # input_patch的大小为torch.Size([512, 512, 4]),而input_patch的类型为<class 'torch.Tensor'>
        input_patch = input_patch.permute(2, 0, 1)
        # print(f"input_patch的大小为{input_patch.shape},而input_patch的类型为{type(input_patch)}") # input_patch的大小为torch.Size([4, 512, 512]),而input_patch的类型为<class 'torch.Tensor'>
        
        # 将NumPy数组gt_patch转换为PyTorch张量    
        gt_patch = torch.from_numpy(gt_patch)
        # print(f"gt_patch的大小为{gt_patch.shape},而gt_patch的类型为{type(gt_patch)}") # gt_patch的大小为torch.Size([1, 1024, 1024, 3]),而gt_patch的类型为<class 'torch.Tensor'>
        # 用于从张量中移除尺寸为1的维度
        gt_patch = torch.squeeze(gt_patch)
        # print(f"gt_patch的大小为{gt_patch.shape},而gt_patch的类型为{type(gt_patch)}") # gt_patch的大小为torch.Size([1024, 1024, 3]),而gt_patch的类型为<class 'torch.Tensor'>
        gt_patch = gt_patch.permute(2, 0, 1)
        # print(f"gt_patch的大小为{gt_patch.shape},而gt_patch的类型为{type(gt_patch)}") # gt_patch的大小为torch.Size([3, 1024, 1024]),而gt_patch的类型为<class 'torch.Tensor'>
        # print("*"*50)
        
        return input_patch, gt_patch, id, ratio  
      
class SonyValDataset(Dataset):
    def __init__(self, file_path, ps=512) -> None:
        super().__init__()
        self.ps = ps
        self.image_pairs = []  # 存储图像路径对和比率的列表
        
        # 读取文件并获取数据
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:  # 仅获取前两个数据
                parts = line.strip().split()
                short_path = parts[0]
                long_path = parts[1]
                short_time_info = float(short_path.split("_")[-1].split("s")[0])  # 将时间信息转换为浮点数
                long_time_info = float(long_path.split("_")[-1].split("s")[0])  # 将时间信息转换为浮点数
                ratio = min(long_time_info / short_time_info, 300)
                self.image_pairs.append((short_path, long_path, ratio))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        short_path, long_path, ratio = self.image_pairs[index]
        # in_fn = os.path.basename(short_path)
        # gt_fn = os.path.basename(long_path)
        
        raw = rawpy.imread(short_path)
        input_image = np.expand_dims(pack_raw(raw), axis=0) * ratio
        gt_raw = rawpy.imread(long_path)
        gt_image = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_image = np.expand_dims(np.float32(gt_image / 65535.0), axis=0)
        
        H = input_image.shape[1]
        W = input_image.shape[2]
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        input_patch = input_image[:, yy:yy + self.ps, xx:xx + self.ps, :]
        gt_patch = gt_image[:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]
    
        # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        # print("*"*50)
        
        # 将NumPy数组input_patch转换为PyTorch张量
        input_patch = torch.from_numpy(input_patch)
        # print(f"input_patch的大小为{input_patch.shape},而input_patch的类型为{type(input_patch)}") # input_patch的大小为torch.Size([1, 512, 512, 4]),而input_patch的类型为<class 'torch.Tensor'>
        input_patch = torch.squeeze(input_patch)
        # print(f"input_patch的大小为{input_patch.shape},而input_patch的类型为{type(input_patch)}") # input_patch的大小为torch.Size([512, 512, 4]),而input_patch的类型为<class 'torch.Tensor'>
        input_patch = input_patch.permute(2, 0, 1)
        # print(f"input_patch的大小为{input_patch.shape},而input_patch的类型为{type(input_patch)}") # input_patch的大小为torch.Size([4, 512, 512]),而input_patch的类型为<class 'torch.Tensor'>
        
        # 将NumPy数组gt_patch转换为PyTorch张量    
        gt_patch = torch.from_numpy(gt_patch)
        # print(f"gt_patch的大小为{gt_patch.shape},而gt_patch的类型为{type(gt_patch)}") # gt_patch的大小为torch.Size([1, 1024, 1024, 3]),而gt_patch的类型为<class 'torch.Tensor'>
        # 用于从张量中移除尺寸为1的维度
        gt_patch = torch.squeeze(gt_patch)
        # print(f"gt_patch的大小为{gt_patch.shape},而gt_patch的类型为{type(gt_patch)}") # gt_patch的大小为torch.Size([1024, 1024, 3]),而gt_patch的类型为<class 'torch.Tensor'>
        gt_patch = gt_patch.permute(2, 0, 1)
        # print(f"gt_patch的大小为{gt_patch.shape},而gt_patch的类型为{type(gt_patch)}") # gt_patch的大小为torch.Size([3, 1024, 1024]),而gt_patch的类型为<class 'torch.Tensor'>
        # print("*"*50)
        
        return input_patch, gt_patch, ratio,index

class SonyEvalDataset(Dataset):
    def __init__(self, file_path, ps=512) -> None:
        super().__init__()
        self.ps = ps
        self.image_pairs = []  # 存储图像路径对和比率的列表
        
        # 读取文件并获取数据
        with open(file_path, 'r') as file:
            lines = file.readlines()
            for line in lines:  # 仅获取前两个数据
                parts = line.strip().split()
                short_path = parts[0]
                long_path = parts[1]
                short_time_info = float(short_path.split("_")[-1].split("s")[0])  # 将时间信息转换为浮点数
                long_time_info = float(long_path.split("_")[-1].split("s")[0])  # 将时间信息转换为浮点数
                ratio = min(long_time_info / short_time_info, 300)
                self.image_pairs.append((short_path, long_path, ratio))

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, index):
        short_path, long_path, ratio = self.image_pairs[index]
        # in_fn = os.path.basename(short_path)
        # gt_fn = os.path.basename(long_path)
        
        raw = rawpy.imread(short_path)
        input_image = np.expand_dims(pack_raw(raw), axis=0) * ratio
        gt_raw = rawpy.imread(long_path)
        gt_image = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        gt_image = np.expand_dims(np.float32(gt_image / 65535.0), axis=0)
        
        H = input_image.shape[1]
        W = input_image.shape[2]
        xx = np.random.randint(0, W - self.ps)
        yy = np.random.randint(0, H - self.ps)
        input_patch = input_image[:, yy:yy + self.ps, xx:xx + self.ps, :]
        gt_patch = gt_image[:, yy * 2:yy * 2 + self.ps * 2, xx * 2:xx * 2 + self.ps * 2, :]
    
        # data augmentation
        # random flip vertically
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=1).copy()
            gt_patch = np.flip(gt_patch, axis=1).copy()
        # random flip horizontally
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2).copy()
            gt_patch = np.flip(gt_patch, axis=2).copy()
        # random transpose
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.transpose(input_patch, (0, 2, 1, 3)).copy()
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3)).copy()

        input_patch = np.minimum(input_patch, 1.0)
        # print("*"*50)
        
        # 将NumPy数组input_patch转换为PyTorch张量
        input_patch = torch.from_numpy(input_patch)
        # print(f"input_patch的大小为{input_patch.shape},而input_patch的类型为{type(input_patch)}") # input_patch的大小为torch.Size([1, 512, 512, 4]),而input_patch的类型为<class 'torch.Tensor'>
        input_patch = torch.squeeze(input_patch)
        # print(f"input_patch的大小为{input_patch.shape},而input_patch的类型为{type(input_patch)}") # input_patch的大小为torch.Size([512, 512, 4]),而input_patch的类型为<class 'torch.Tensor'>
        input_patch = input_patch.permute(2, 0, 1)
        # print(f"input_patch的大小为{input_patch.shape},而input_patch的类型为{type(input_patch)}") # input_patch的大小为torch.Size([4, 512, 512]),而input_patch的类型为<class 'torch.Tensor'>
        
        # 将NumPy数组gt_patch转换为PyTorch张量    
        gt_patch = torch.from_numpy(gt_patch)
        # print(f"gt_patch的大小为{gt_patch.shape},而gt_patch的类型为{type(gt_patch)}") # gt_patch的大小为torch.Size([1, 1024, 1024, 3]),而gt_patch的类型为<class 'torch.Tensor'>
        # 用于从张量中移除尺寸为1的维度
        gt_patch = torch.squeeze(gt_patch)
        # print(f"gt_patch的大小为{gt_patch.shape},而gt_patch的类型为{type(gt_patch)}") # gt_patch的大小为torch.Size([1024, 1024, 3]),而gt_patch的类型为<class 'torch.Tensor'>
        gt_patch = gt_patch.permute(2, 0, 1)
        # print(f"gt_patch的大小为{gt_patch.shape},而gt_patch的类型为{type(gt_patch)}") # gt_patch的大小为torch.Size([3, 1024, 1024]),而gt_patch的类型为<class 'torch.Tensor'>
        # print("*"*50)
        
        return input_patch, gt_patch, ratio,index


# 创建数据集实例
if __name__ == '__main__':
    
    # eval_list
    input_dir = '/raid/hbj/data/Sony/short/'
    gt_dir = '/raid/hbj/data/Sony/long/'
    trainset = SonyDataset(input_dir, gt_dir, ps=512)
    dataset_size = len(trainset)
    print(dataset_size) # 598
    # 选择一个索引
    index = 2
    input_patch, scale_patch, gt_patch, index, ratio = trainset[index]
    print(f"input_patch的大小为{input_patch.shape},类型为{type(input_patch)}") # 
    # input_patch的大小为torch.Size([4, 512, 512]),类型为<class 'torch.Tensor'>
    print(f"scale_patch的大小为{scale_patch.shape},类型为{type(scale_patch)}") 
    # scale_patch的大小为torch.Size([3, 1024, 1024]),类型为<class 'torch.Tensor'>
    print(f"gt_patch的大小为{gt_patch.shape},类型为{type(gt_patch)}") 
    # gt_patch的大小为torch.Size([3, 1024, 1024]),类型为<class 'torch.Tensor'>
    
    print(f"曝光时间为{ratio}")
    print(f"索引为{index}")