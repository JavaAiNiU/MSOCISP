import numpy as np
from matplotlib import pyplot as plt
import torch
from skimage.metrics import structural_similarity as ssim
import torch.nn as nn
import numpy as np
import os
import torchvision.transforms.functional as TF
import torch.nn.functional as F

def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out

def create_or_check_file(file_path):
    if os.path.exists(file_path):
        print("文件已存在")
    else:
        # 文件不存在，创建文件
        os.makedirs(file_path)
        print("文件已成功创建")

def randomTrue():
    isTrue = np.random.randint(2, size=1)[0] == 1
    return isTrue

def PSNR_Loss(low, high):
    shape = low.shape
    if len(shape) <= 3:
        psnr = -10.0 * torch.log(torch.mean(torch.pow(high-low, 2))) / torch.log(torch.as_tensor(10.0))
    else:
        psnr = torch.zeros(shape[0])
        for i in range(shape[0]):
            psnr[i]=-10.0 * torch.log(torch.mean(torch.pow(high[i]-low[i], 2))) / torch.log(torch.as_tensor(10.0))
        # print(psnr)
        psnr = torch.mean(psnr)# / shape[0]
    return psnr 

def psnrValue(inp, avgOut):
    # 函数接受两个输入参数，inp和avgOut，它们都是四维张量（Tcnt，Ch，Hig，Wid）。
    # 具体来说，inp包含原始图像的信息，而avgOut包含经过处理后的图像信息。
    totalPsnr = 0 
    Tcnt, Ch, Hig, Wid = inp.shape

    for i in range(Tcnt):
        # 通过使用torch.clamp函数将avgOut中的值限制在0到1之间，确保图像的像素值在有效范围内
        avgOut[i] = torch.clamp(avgOut[i], min=0.0, max=1.0)
        # 计算均方误差（MSE），表示原始图像与经过处理后的图像之间的差异
        mse = torch.sum((inp[i] - avgOut[i])**2)/(Ch*Hig*Wid)
        # 使用MSE计算峰值信噪比（PSNR），并将其累积到totalPsnr中
        psnr =  -10*torch.log10(mse)
        totalPsnr += psnr

    AvgPsnr = totalPsnr/Tcnt
    return AvgPsnr  

def CalcLoss(Im1, Im2):
    lossval = torch.mean(torch.abs(Im1 - Im2))
    return lossval

def ssim_computer(img1, img2):
    """
    计算图像的结构相似性指数（SSIM）
    
    参数：
    - img1: 第一个图像(torch.Tensor)
    - img2: 第二个图像(torch.Tensor)
    
    返回：
    - ssim_value: 结构相似性指数
    """
    # 去除维度为1的维度
    img1 = img1.squeeze(0)
    img2 = img2.squeeze(0)
    
    # 将图像的像素值缩放到[0, 1]范围
    img1 = (img1 - img1.min()) / (img1.max() - img1.min())
    img2 = (img2 - img2.min()) / (img2.max() - img2.min())

   # 将PyTorch张量转换为NumPy数组
    img1_np = img1.permute(1, 2, 0).detach().cpu().numpy()
    img2_np = img2.permute(1, 2, 0).detach().cpu().numpy()

    # 计算SSIM
    ssim_value, _ = ssim(img1_np, img2_np, multichannel=True, full=True)

    return ssim_value

def calculate_psnr(real_images, output_images):
    # 计算均方误差（MSE）
    mse = torch.mean((real_images - output_images) ** 2)
    
    # 如果 MSE 为零，表示两个张量完全相同，PSNR 为正无穷
    if mse == 0:
        return float('inf')
    
    # 计算 PSNR
    max_val = 1.0  # 假设张量的像素值范围在 [0, 1] 之间
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def calculate_ssim(tensor1, tensor2):
    """
    Calculate Structural Similarity Index (SSIM) between two tensors.

    Args:
        tensor1 (torch.Tensor): First input tensor.
        tensor2 (torch.Tensor): Second input tensor.

    Returns:
        float: SSIM value.
    """
    # Ensure tensors are of the same type
    if tensor1.dtype != tensor2.dtype:
        raise ValueError("Input tensors must have the same data type")

    # Calculate SSIM
    ssim_value = F.mse_loss(tensor1, tensor2)
    
    return ssim_value.item()

def cal_psnr(real_images, output_images):
    mse = torch.mean((real_images - output_images) ** 2, axis=(1,2,3))
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return psnr

def cal_ssim(real_images, output_images):
    batch_size = real_images.shape[0]
    ssim_values = []
    for i in range(batch_size):
        real_image_pil = TF.to_pil_image(real_images[i])
        output_image_pil = TF.to_pil_image(output_images[i])
        real_image_gray = TF.to_grayscale(real_image_pil)
        output_image_gray = TF.to_grayscale(output_image_pil)
        real_image_np = np.array(real_image_gray)
        output_image_np = np.array(output_image_gray)
        ssim_value = ssim(real_image_np, output_image_np)
        ssim_values.append(ssim_value)
    return np.array(ssim_values)

def Sobel_loss(output_img, target_img):
    # 计算灰度图像
    output_gray = torch.mean(output_img, dim=1, keepdim=True)
    target_gray = torch.mean(target_img, dim=1, keepdim=True)
    
    # 使用Sobel滤波器计算图像的边缘
    sobel_filter_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(output_img.device)
    sobel_filter_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(output_img.device)

    output_grad_x = F.conv2d(output_gray, sobel_filter_x)
    output_grad_y = F.conv2d(output_gray, sobel_filter_y)

    target_grad_x = F.conv2d(target_gray, sobel_filter_x)
    target_grad_y = F.conv2d(target_gray, sobel_filter_y)
    
    # 计算Sobel损失
    sobel_loss = torch.mean(torch.abs(output_grad_x - target_grad_x) + torch.abs(output_grad_y - target_grad_y))
    
    return sobel_loss



class MS_SSIM_Loss(torch.nn.Module):
    def __init__(self, data_range=1.0, size_average=True):
        super(MS_SSIM_Loss, self).__init__()
        self.data_range = data_range
        self.size_average = size_average
        self.levels = 5
        self.weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    
    def forward(self, output_img, target_img):
        output_img = output_img / self.data_range
        target_img = target_img / self.data_range

        ssim_value = torch.zeros(1).to(output_img.device)
        for level in range(self.levels):
            ssim_level = self._ssim(output_img, target_img)
            ssim_value += self.weights[level] * ssim_level
            if level < self.levels - 1:
                output_img = F.avg_pool2d(output_img, kernel_size=2)
                target_img = F.avg_pool2d(target_img, kernel_size=2)

        ms_ssim_loss = 1 - ssim_value

        if self.size_average:
            ms_ssim_loss = ms_ssim_loss.mean()

        return ms_ssim_loss

    def _ssim(self, img1, img2, window_size=11, sigma=1.5):
        K1 = 0.01
        K2 = 0.03
        L = self.data_range
        C1 = (K1 * L) ** 2
        C2 = (K2 * L) ** 2

        mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
        mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return ssim_map.mean()

def MS_ssim_loss(output_img, target_img):
    data_range=1.0
    size_average=True
    levels = 5  # 多尺度层数
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])  # 权重系数
    
    # 将图像归一化到[0, 1]
    output_img = output_img / data_range
    target_img = target_img / data_range
    
    ssim_value = torch.zeros(1).to(output_img.device)
    for level in range(levels):
        ssim_level = _ssim(output_img, target_img, data_range=data_range)
        ssim_value += weights[level] * ssim_level
        if level < levels - 1:
            output_img = F.avg_pool2d(output_img, kernel_size=2)
            target_img = F.avg_pool2d(target_img, kernel_size=2)

    ms_ssim_loss = 1 - ssim_value

    if size_average:
        ms_ssim_loss = ms_ssim_loss.mean()
    
    return ms_ssim_loss

def _ssim(img1, img2, data_range=1.0, window_size=11, sigma=1.5):
    K1 = 0.01
    K2 = 0.03
    L = data_range  # 数据范围（例如：像素值范围为[0, 1]时，L=1.0）
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()




# 用于打印模型参数数量的辅助函数
def print_model_parm_nums(model):
    """函数接受一个模型对象作为参数，计算该模型中的所有参数的总数量"""
    # 过遍历模型的所有参数（通过 model.parameters() 获得）
    # 并使用 param.nelement() 获取每个参数的元素数量，最后将这些数量求和得到总的参数数量
    total = sum([param.nelement() for param in model.parameters()])
    print('  + Number of params: %.2fM' % (total / 1e6))
    
    
    
    
# 保存实时指标数据到指定文件夹
def save_metrics(epoch, loss, psnr, ssim, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    np.savez(os.path.join(save_folder, "metrics.npz"), epoch=epoch, loss=loss, psnr=psnr, ssim=ssim)

# 加载实时指标数据
def load_metrics(load_folder):
    data = np.load(os.path.join(load_folder, "metrics.npz"))
    return data["epoch"], data["loss"], data["psnr"], data["ssim"]

# 绘制损失、PSNR 和 SSIM 图像并保存
def plot_metrics(epoch, loss, psnr, ssim, save_folder=None):
    plt.figure(figsize=(12, 4))

    # 绘制损失图像
    plt.subplot(1, 3, 1)
    plt.plot(epoch, loss, label="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, "loss_plot.png"))
    else:
        plt.show()

    # 绘制PSNR图像
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epoch, psnr, label="PSNR")
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.title("PSNR")
    plt.grid(True)
    plt.legend()
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, "psnr_plot.png"))
    else:
        plt.show()

    # 绘制SSIM图像
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epoch, ssim, label="SSIM")
    plt.xlabel("Epoch")
    plt.ylabel("SSIM")
    plt.title("SSIM")
    plt.grid(True)
    plt.legend()
    if save_folder:
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        plt.savefig(os.path.join(save_folder, "ssim_plot.png"))
    else:
        plt.show()

if __name__ == '__main__':

    # 示例用法
    img1 = torch.randn(4, 3, 1024, 1024)
    img2 = torch.randn(4, 3, 1024, 1024)

    ssim_value = cal_ssim(img1, img2) 
    psnr_value = cal_psnr(img1, img2)
    print("PSNR:", psnr_value)   # PSNR: tensor([-3.0061, -3.0085, -3.0109, -3.0149]) 
    print("SSIM:", ssim_value) # SSIM: [0.01114748 0.01245222 0.01350998 0.01128451]
    
    # Example usage:
    net_output = torch.rand(16, 3, 1024, 1024)  # Example random tensor
    real_image = torch.rand(16, 3, 1024, 1024)   # Example random tensor

    ssim_value = calculate_ssim(net_output, real_image)
    print("SSIM:", ssim_value) # SSIM: 0.16664151847362518
    psnr_value = calculate_psnr(img1, img2)
    print("PSNR:", psnr_value)   # PSNR: -3.0100979804992676
