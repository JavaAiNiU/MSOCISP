import torchvision.models as models
import torch.nn as nn
import torch

class VGGPerceptualLoss(nn.Module):
    def __init__(self, feature_layer=15):
        """
        feature_layer=15 对应 VGG16 中的 relu3_3 层。
        """
        super(VGGPerceptualLoss, self).__init__()
        # 加载预训练的VGG16模型，并截取到指定的特征层
        vgg = models.vgg16(pretrained=True).features[:feature_layer+1].eval()
        # 冻结VGG参数，不参与反向传播
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.criterion = nn.L1Loss()
        
        # ImageNet 归一化参数，用于对齐VGG的预训练输入分布
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        # 确保输入在 [0, 1] 范围
        x = torch.clip(x, 0, 1.0)
        y = torch.clip(y, 0, 1.0)
        
        # 归一化
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        
        # 提取特征并计算特征图之间的 L1 损失
        x_vgg = self.vgg(x)
        y_vgg = self.vgg(y)
        return self.criterion(x_vgg, y_vgg)