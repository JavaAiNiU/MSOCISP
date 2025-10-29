from turtle import forward
import torch
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from timm.layers import DropPath, to_2tuple, trunc_normal_
import math

import matplotlib.pyplot as plt

# 新增：特征图可视化函数
def visualize_compressed_feature(feature_map, num_compress=4, save_path="compressed_mso_rg.png"):
    """
    将多通道特征图压缩为指定数量的图像并可视化
    feature_map: 输入特征图，形状为 [B, C, H, W]（通常 B=1）
    num_compress: 压缩后的图像数量（如4）
    save_path: 保存路径
    """
    # 只取第一个样本（B=0）
    feature_map = feature_map[0].cpu().detach().numpy()  # 形状：[C, H, W]
    C, H, W = feature_map.shape
    
    # 检查通道数是否能被压缩数量整除（这里48通道 / 4 = 12通道/组）
    assert C % num_compress == 0, f"通道数 {C} 不能被 {num_compress} 整除，请调整参数"
    channels_per_group = C // num_compress  # 每组通道数：48//4=12
    
    # 初始化压缩后的特征图列表
    compressed_maps = []
    for i in range(num_compress):
        # 提取第i组通道（如第0组：0-11通道，第1组：12-23通道...）
        start = i * channels_per_group
        end = (i + 1) * channels_per_group
        group = feature_map[start:end, :, :]  # 形状：[12, H, W]
        
        # 组内压缩（求和或平均，保留特征整体强度）
        compressed = np.mean(group, axis=0)  # 形状：[H, W]（也可使用np.mean）
        
        # 归一化到0-255（便于显示）
        compressed = (compressed - compressed.min()) / (compressed.max() - compressed.min() + 1e-8)
        compressed = (compressed * 255).astype(np.uint8)
        compressed_maps.append(compressed)
    
    # 可视化4张压缩后的特征图
    plt.figure(figsize=(12, 10))
    for i in range(num_compress):
        plt.subplot(2, 2, i + 1)  # 2行2列网格
        plt.imshow(compressed_maps[i], cmap="plasma")  # 用plasma颜色映射增强细节
        plt.title(f"Compressed Group {i+1} (Channels {i*channels_per_group + 1}-{(i+1)*channels_per_group})")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"压缩后的特征图已保存至 {save_path}")


# 定义了一个离散小波变换（Discrete Wavelet Transform, DWT）和逆离散小波变换（Inverse DWnw）的操作，它们用于对图像进行小波变换和逆变换
# 小波变换是一种在信号和图像处理中常用的技术，用于分析信号和图像的不同频率成分
def dwt_init(x):
    """
    将输入的图像张量 x 分解为四个子图像,分别代表小波变换的低频部分(LL)、水平高频部分(HL)、垂直高频部分(LH)、和对角高频部分(HH)。
    这些子图像经过适当的加权和相加操作后，返回一个包含这四个部分的张量
    """
    # x01 和 x02 是 x 的两个子图像，它们分别包含了 x 中的偶数行和奇数行
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    # x1 包含了 x01 中的偶数列，而 x2 包含了 x02 中的偶数列
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    # x3 包含了 x01 中的奇数列，而 x4 包含了 x02 中的奇数列
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    # 低频部分
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4
    # 函数返回一个张量，其中包含了上述四个部分，它们在深度维度上连接在一起
    """
    如果x_LL、x_HL、x_LH、x_HH都是形状为(batch_size, num_channels, height, width)的张量，
    那么使用torch.cat((x_LL, x_HL, x_LH, x_HH), 1)将它们连接在一起，
    得到一个形状为(batch_size, 4 * num_channels, height, width)的张量。
    """
    return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)

def iwt_init(x):
    """
    用于执行逆离散小波变换，将四个子图像合并还原成原始图像。
    它接受一个包含四个小波变换部分的输入张量，然后执行逆变换操作，返回还原后的原始图像。
    """
    r = 2
    # 
    in_batch, in_channel, in_height, in_width = x.size()
    out_batch, out_channel, out_height, out_width = in_batch, int(
        in_channel / (r**2)), r * in_height, r * in_width
    x1 = x[:, 0:out_channel, :, :] / 2
    x2 = x[:, out_channel:out_channel * 2, :, :] / 2
    x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
    x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
    h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().to(x.device)
    # 将四个子图像的信息合并还原成原始图像h
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4

    return h

class DWT(nn.Module):
    """
    离散小波变换的 PyTorch 模块，它继承自 nn.Module。在其 forward 方法中，它调用了 dwt_init 函数执行小波变换操作，并返回变换后的图像
    """
    def __init__(self):
        super(DWT, self).__init__()
        # 该模块的参数不会进行梯度计算，因为小波变换操作是固定的
        self.requires_grad = False

    def forward(self, x):
        # 执行离散小波变换操作，并将变换后的图像作为结果返回
        return dwt_init(x)

class IWT(nn.Module):
    """执行逆离散小波变换：执行逆变换操作，并返回还原后的图像"""
    def __init__(self):
        super(IWT, self).__init__()
        # 该模块的参数不会进行梯度计算，因为小波变换操作是固定的
        self.requires_grad = False

    def forward(self, x):
        # 执行逆离散小波变换操作，将还原后的图像作为结果返回
        return iwt_init(x)
    

# 激活函数
class LeakyReLU(nn.Module):
    def __init__(self):
        super(LeakyReLU, self).__init__()
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(x)

# 双层卷积 
class UNetConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNetConvBlock, self).__init__()
        self.UNetConvBlock = torch.nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1),
            LeakyReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1),
            LeakyReLU()
        )
    def forward(self, x):
        return self.UNetConvBlock(x)

class CALayer(nn.Module):
    def __init__(self,in_ch,reduction=16):
        super(CALayer,self).__init__()
        self.a = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch,in_ch // reduction,1),
            nn.Conv2d(in_ch // reduction,in_ch,1),
            nn.Sigmoid(),
        )
        
    def forward(self,x):
        return x * self.a(x)

class RCAB(nn.Module):
    def __init__(self,in_ch,reduction=16):
        super(RCAB, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(in_ch,in_ch,3,padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_ch,in_ch,3,padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            CALayer(in_ch,reduction),
        #nn.LeakyReLU(0.2, inplace=True),
        )
        
    def forward(self,x):
        res = self.res(x) + x
        return res
    
class RG(nn.Module):
    def __init__(self, in_ch, reduction=16, num_rcab=2):
        super(RG, self).__init__()
        self.layers = nn.Sequential(
            *[RCAB(in_ch, reduction) for _ in range(num_rcab)]
        )

    def forward(self, x):
        out = self.layers(x)
        return out + x
    
class StackedRG(nn.Module):
    def __init__(self, in_ch, reduction=16, num_rcab=2, num_rg=4):
        super(StackedRG, self).__init__()
        self.stack = nn.Sequential( nn.Conv2d(64, 64, 3, 1, 1), LeakyReLU(),
                                    nn.Conv2d(64, 64, 3, 1, 1), LeakyReLU(),
                                    nn.Conv2d(64, 64, 3, 1, 1), LeakyReLU(),
                                    nn.Conv2d(64, 64, 3, 1, 1), LeakyReLU())
        
    def forward(self, x):
        out = self.stack(x)
        return out   # + x
    
def split_feature_map(feature_map, num_groups=4):
    input_channels = feature_map.size(1)
    group_size = input_channels // num_groups
    groups = []
    for i in range(num_groups):
        start_channel = i * group_size
        end_channel = (i + 1) * group_size
        group = feature_map[:, start_channel:end_channel, :, :]
        groups.append(group)
    LL = groups[0]
    LH = groups[1]
    HL = groups[2]
    HH = groups[3]
    return LL,LH,HL,HH

class High_Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = UNetConvBlock(12, 64) 
        self.conv2 = UNetConvBlock(64, 128)
        self.conv3 = UNetConvBlock(128, 256)
        # self.conv4 = UNetConvBlock(128, 256)
        
    def forward(self,x):
        H1 = self.conv1(x)  
        # print(f"H1的大小为{H1.shape},而它的类型为{type(H1)}") # conv1的大小为torch.Size([1, 64, 256, 256]),而它的类型为<class 'torch.Tensor'>
        pool1 = F.max_pool2d(H1,kernel_size=2)  
        # print(f"H1的大小为{H1.shape},而它的类型为{type(H1)}") # H1的大小为torch.Size([1, 64, 128, 128]),而它的类型为<class 'torch.Tensor'>
        
        H2 = self.conv2(pool1)  
        # print(f"H2的大小为{H2.shape},而它的类型为{type(H2)}") # conv2的大小为torch.Size([1, 64, 128, 128]),而它的类型为<class 'torch.Tensor'>
        pool2 = F.max_pool2d(H2,kernel_size=2)  
        # print(f"pool2的大小为{pool2.shape},而它的类型为{type(pool2)}") # pool2的大小为torch.Size([1, 64, 64, 64]),而它的类型为<class 'torch.Tensor'>
        
        H3 = self.conv3(pool2)  
        # print(f"H3的大小为{H3.shape},而它的类型为{type(H3)}") # conv3的大小为torch.Size([1, 128, 64, 64]),而它的类型为<class 'torch.Tensor'>
        
        H4 = F.max_pool2d(H3,kernel_size=2)  
        # print(f"H3的大小为{H3.shape},而它的类型为{type(H3)}") # H3的大小为torch.Size([1, 128, 32, 32]),而它的类型为<class 'torch.Tensor'>
        
        return H1,H2,H3,H4
    
class Low_Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = UNetConvBlock(4, 64)  # We have 4 Channel (R,G,B,G)- Bayer Pattern Input
        self.conv2 = UNetConvBlock(64, 128)
        self.conv3 = UNetConvBlock(128, 256)
        # self.conv4 = UNetConvBlock(128, 256)
        
    def forward(self,x):
        L1 = self.conv1(x)  
        # print(f"conv1的大小为{conv1.shape},而它的类型为{type(conv1)}") # conv1的大小为torch.Size([1, 32, 256, 256]),而它的类型为<class 'torch.Tensor'>
        pool1 = F.max_pool2d(L1,kernel_size=2)  
        # print(f"H1的大小为{H1.shape},而它的类型为{type(H1)}") # H1的大小为torch.Size([1, 32, 128, 128]),而它的类型为<class 'torch.Tensor'>
        
        L2 = self.conv2(pool1)  
        # print(f"conv2的大小为{conv2.shape},而它的类型为{type(conv2)}") # conv2的大小为torch.Size([1, 64, 128, 128]),而它的类型为<class 'torch.Tensor'>
        pool2 = F.max_pool2d(L2,kernel_size=2)  
        # print(f"pool2的大小为{pool2.shape},而它的类型为{type(pool2)}") # pool2的大小为torch.Size([1, 64, 64, 64]),而它的类型为<class 'torch.Tensor'>
        
        L3 = self.conv3(pool2)  
        # print(f"conv3的大小为{conv3.shape},而它的类型为{type(conv1)}") # conv3的大小为torch.Size([1, 128, 64, 64]),而它的类型为<class 'torch.Tensor'>
        L4 = F.max_pool2d(L3,kernel_size=2)  
        # print(f"H3的大小为{H3.shape},而它的类型为{type(H3)}") # H3的大小为torch.Size([1, 128, 32, 32]),而它的类型为<class 'torch.Tensor'>
        
        # L4 = self.conv4(pool3)  
        # # print(f"conv4的大小为{conv4.shape},而它的类型为{type(conv4)}") # conv4的大小为torch.Size([1, 256, 32, 32]),而它的类型为<class 'torch.Tensor'>
        
        # poolL = F.max_pool2d(L4,kernel_size=2)  
        # # print(f"poolL的大小为{poolL.shape},而它的类型为{type(poolL)}") # poolL的大小为torch.Size([1, 256, 16, 16]),而它的类型为<class 'torch.Tensor'>
        return L1,L2,L3,L4
    
# 定义空间注意模块
class Spatial_Attention(nn.Module):
    def __init__(self):
        super(Spatial_Attention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return x * self.sigmoid(x)

# class Spatial_Attention(nn.Module):
#     def __init__(self, in_channels):
#         super(Spatial_Attention, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv2d(in_channels, 1, kernel_size=1)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.sigmoid(out)
#         # print(f"out的大小为{out.shape}")
#         return out
    
#交叉注意力模块、
class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        B, N, C = x.shape
        tx = x.transpose(1, 2).view(B, C, H, W)
        conv_x = self.dwconv(tx)
        return conv_x.flatten(2).transpose(1, 2)
    
class MixFFN_skip(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.fc1 = nn.Linear(c1, c2)
        self.dwconv = DWConv(c2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(c2, c1)
        self.norm1 = nn.LayerNorm(c2)
        self.norm2 = nn.LayerNorm(c2)
        self.norm3 = nn.LayerNorm(c2)

    def forward(self, x: torch.Tensor, H, W) -> torch.Tensor:
        ax = self.act(self.norm1(self.dwconv(self.fc1(x), H, W) + self.fc1(x)))
        out = self.fc2(ax)
        return out
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) #对应Squeeze操作
        y = self.fc(y).view(b, c, 1, 1) #对应Excitation操作
        return x * y.expand_as(x)
    
class Cross_Attention(nn.Module):
    def __init__(self, key_channels, value_channels, head_count=1):
        super().__init__()
        self.key_channels = key_channels
        self.head_count = head_count
        self.value_channels = value_channels
        self.reprojection = nn.Conv2d(value_channels, 2 * value_channels, 1)
        self.norm = nn.LayerNorm(2 * value_channels)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.25]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.25]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.25]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.25]), requires_grad=True)
        # self.attn5 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    # x2 should be higher-level representation than x1
    #比较的值在前，主干在后
    def forward(self, x1, x2, H, W):
        B, N, D = x1.size()  # (Batch, Tokens, Embedding dim)
        # Re-arrange into a (Batch, Embedding dim, Tokens)
        # 代码写反了,V实际是Q,Q实际是V
        keys = x2.transpose(1, 2)
        queries = x2.transpose(1, 2)
        values = x1.transpose(1, 2)
        
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count

        attended_values = []
        for i in range(self.head_count):
            
            key = F.softmax(keys[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels : (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels : (i + 1) * head_value_channels, :]

            context = key @ value.transpose(1, 2)  # dk*dv

            # print("key:",key.shape)
            # print("value.transpose(1, 2):",value.transpose(1, 2).shape)
            # print("context:",context.shape)
            
            mask1 = torch.zeros(B, D, D, device=x1.device, requires_grad=False)
            mask2 = torch.zeros(B, D, D, device=x1.device, requires_grad=False)
            mask3 = torch.zeros(B, D, D, device=x1.device, requires_grad=False)
            mask4 = torch.zeros(B, D, D, device=x1.device, requires_grad=False)

            index = torch.topk(context, k=int(D * 1 / 2), dim=-1, largest=True)[1]
            mask1.scatter_(-1, index, 1.)
            attn1 = torch.where(mask1 > 0, context, torch.full_like(context, float('-inf')))

            index = torch.topk(context, k=int(D * 2 / 3), dim=-1, largest=True)[1]
            mask2.scatter_(-1, index, 1.)
            attn2 = torch.where(mask2 > 0, context, torch.full_like(context, float('-inf')))

            index = torch.topk(context, k=int(D * 3 / 4), dim=-1, largest=True)[1]
            mask3.scatter_(-1, index, 1.)
            attn3 = torch.where(mask3 > 0, context, torch.full_like(context, float('-inf')))

            index = torch.topk(context, k=int(D * 4 / 5), dim=-1, largest=True)[1]
            mask4.scatter_(-1, index, 1.)
            attn4 = torch.where(mask4 > 0, context, torch.full_like(context, float('-inf')))

            # attn5 = maskv

            attn1 = attn1.softmax(dim=-1)
            attn2 = attn2.softmax(dim=-1)
            attn3 = attn3.softmax(dim=-1)
            attn4 = attn4.softmax(dim=-1)
            # attn5 = attn5.softmax(dim=-1)


            # print("attn1:",attn1.shape)
            # print("query:",query.shape)
            out1 = (attn1 @ query)
            # print("out1:",out1.shape)
            out2 = (attn2 @ query)
            out3 = (attn3 @ query)
            out4 = (attn4 @ query)
            # out5 = (attn5 @ query)

            attended_value = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4 #+ out5 * self.attn5
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1).reshape(B, D, H, W)
        # print("aggregated_values:",aggregated_values.shape)
        reprojected_value = self.reprojection(aggregated_values).reshape(B, 2 * D, N).permute(0, 2, 1)
        # print("reprojected_value1:",reprojected_value.shape)     
        reprojected_value = self.norm(reprojected_value)
        # print("reprojected_value2:",reprojected_value.shape)     

        return reprojected_value
    
class CrossAttentionBlock(nn.Module):
    """
    Input ->    x1:[B, N, D] - N = H*W
                x2:[B, N, D]
    Output -> y:[B, N, D]
    D is half the size of the concatenated input (x1 from a lower level and x2 from the skip connection)
    """
    def __init__(self, in_dim, key_dim, value_dim,head_count=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_dim)
        self.attn = Cross_Attention(key_dim, value_dim, head_count=head_count)
        # self.norm2 = nn.LayerNorm((in_dim*2))
        # self.mlp = MixFFN_skip(int(in_dim*2) , int(in_dim * 4))
        # self.channel_att = SELayer(channel=int(in_dim*2))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        B,H,W,C = x1.shape
        x1 = x1.reshape(B, H*W, C)
        x2 = x2.reshape(B, H*W, C)
        norm_1 = self.norm1(x1)
        norm_2 = self.norm1(x2)
        
        attn = self.attn(norm_1, norm_2, H, W)
        
        # residual = torch.cat([x1, x2], dim=2)
        
        # tx = residual + attn
        # tx = attn
        
        # mx = tx + self.mlp(self.norm2(tx),H,W)
        # cx = mx.view(B, H, W, 2 * C).permute(0, 3, 1, 2)
        attn = attn.view(B, H, W, 2 * C).permute(0, 3, 1, 2)
        # cx = self.channel_att(cx)
        return attn
        
# ViT中使用patch merge，可以轻易地将特征张量缩小。但是此过程最初旨在组合非重叠的图像或特征块，不能保持patch周围的局部连续性。
# 因此SegFormer使用Overlap Patch Merge，进行特征融合从而取得分层的特征。
# 在具体实现中，是使用卷积操作进行的Patch Merge，即通过设计卷积核大小和步幅，来降低特征分辨率。
# 同时卷积应该也实现了Embedding操作，将其投射到给定维度。

# 在SegFormer中，为了保持patch周围之间的局部连续性，使用了重叠的patch。
# 这与之前ViT和PVT做patch embedding时每个patch是独立的不同，这样可以保证局部连续性

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size=3, stride=1, in_chans=3, embed_dim=48):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m): 
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        if self.patch_size[0] == 7:
            x = self.proj(x)
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
        else:
            # x = x.permute(0, 3, 1, 2)
            x = self.proj(x)
            x = x.permute(0, 2, 3, 1)
            x = self.norm(x)
        return x      
   
class GuFilter(nn.Module):
    def __init__(self, kernel_size=5, sigma=3, channels=16):
        """
        高斯滤波模块，支持多通道输入(b,c,h,w)的每个通道独立卷积
        
        参数:
            kernel_size: 高斯核大小（奇数）
            sigma: 高斯函数标准差
            channels: 输入数据的通道数
        """
        super(GuFilter, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels
        
        # 生成各通道共享的高斯核
        self.register_buffer('gaussian_kernel', self._create_gaussian_kernel())
    
    def _create_gaussian_kernel(self):
        """创建符合公式的二维高斯卷积核"""
        # 确保核大小为奇数
        if self.kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")
            
        # 生成坐标网格
        x = torch.arange(-self.kernel_size//2 + 1, self.kernel_size//2 + 1, dtype=torch.float32)
        y = torch.arange(-self.kernel_size//2 + 1, self.kernel_size//2 + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        
        # 计算高斯核: G(x,y;σ) = (1/2πσ²) * exp(-(x²+y²)/2σ²)
        kernel = (1 / (2 * math.pi * self.sigma ** 2)) * \
                 torch.exp(-(xx ** 2 + yy ** 2) / (2 * self.sigma ** 2))
        
        # 归一化处理
        kernel = kernel / torch.sum(kernel)
        print(kernel)
        # 重塑为卷积核格式 [out_channels, in_channels/group, kh, kw]
        return kernel.reshape(1, 1, self.kernel_size, self.kernel_size).repeat(self.channels, 1, 1, 1)
        
    
    def forward(self, x):
        """
        前向传播函数，对多通道输入的每个通道应用高斯卷积
        
        参数:
            x: 输入张量，形状为 (b, c, h, w)
            
        返回:
            卷积后的张量，形状保持 (b, c, h, w)
        """
        # 验证输入通道数与初始化一致
        assert x.shape[1] == self.channels, f"Input channels {x.shape[1]} != initialized channels {self.channels}"
        
        # 应用卷积，使用groups参数确保每个通道独立卷积
        padding = self.kernel_size // 2
        return F.conv2d(
            x, 
            self.gaussian_kernel, 
            padding=padding,
            groups=self.channels
        )
class GuFilter_D(nn.Module):
    def __init__(self, kernel_size=None, initial_sigma=3, channels=16):
        """
        可训练的高斯滤波模块，支持动态生成不同sigma的高斯核
        
        参数:
            kernel_size: 高斯核大小（奇数）
            initial_sigma: sigma的初始值
            channels: 输入数据的通道数
        """
        super(GuFilter_D, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        
        # 将sigma定义为可训练的参数
        self.sigma = nn.Parameter(torch.tensor(initial_sigma, dtype=torch.float32))
        
    def forward(self, x):
        """
        前向传播函数，根据当前sigma动态生成高斯核并应用卷积
        
        参数:
            x: 输入张量，形状为 (b, c, h, w)
            
        返回:
            卷积后的张量，形状保持 (b, c, h, w)
        """
        # 验证输入通道数与初始化一致
        assert x.shape[1] == self.channels, f"Input channels {x.shape[1]} != initialized channels {self.channels}"
           # 自动计算核大小（覆盖99.7%能量）
       
        self.kernel_size = int(6 * self.sigma + 1)
        self.kernel_size = self.kernel_size + 1 if self.kernel_size % 2 == 0 else self.kernel_size
        # 确保核大小为奇数
        if self.kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")
            
        # 生成坐标网格
        x_coords = torch.arange(-self.kernel_size//2 + 1, self.kernel_size//2 + 1, 
                               dtype=torch.float32, device=x.device)
        y_coords = torch.arange(-self.kernel_size//2 + 1, self.kernel_size//2 + 1, 
                               dtype=torch.float32, device=x.device)
        xx, yy = torch.meshgrid(x_coords, y_coords, indexing='ij')
        
        # 计算高斯核: G(x,y;σ) = (1/2πσ²) * exp(-(x²+y²)/2σ²)
        # print("self.sigma:",self.sigma)
        sigma_sq = self.sigma ** 2
        kernel = (1 / (2 * math.pi * sigma_sq)) * \
                 torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_sq))
        
        # 归一化处理
        kernel = kernel / torch.sum(kernel)
        
        # 重塑为卷积核格式 [out_channels, in_channels/group, kh, kw]
        kernel = kernel.reshape(1, 1, self.kernel_size, self.kernel_size)
        kernel = kernel.repeat(self.channels, 1, 1, 1)
        
        # 应用卷积，使用groups参数确保每个通道独立卷积
        padding = self.kernel_size // 2
        return F.conv2d(
            x, 
            kernel, 
            padding=padding,
            groups=self.channels
        )

class MuitlConv(nn.Module):
    """
    保持输入输出通道数一致、特征维度一致的卷积模块
    卷积核大小通过初始化参数传入，激活函数使用LeakyReLU
    """
    def __init__(self, kernel_size, in_channels):
        super(MuitlConv, self).__init__()
        # 计算padding确保输出空间维度与输入一致（stride=1时）
        padding = kernel_size // 2
        # 定义卷积层：输入输出通道一致，步幅为1，padding保证维度不变
        self.stack = nn.Sequential( nn.Conv2d(in_channels, in_channels, kernel_size, 1, padding), LeakyReLU(),
                                    nn.Conv2d(in_channels, in_channels, kernel_size, 1, padding), LeakyReLU())

    def forward(self, x):
        """
        前向传播：卷积 -> LeakyReLU
        Args:
            x: 输入特征，形状为[B, C, H, W]
        Returns:
            输出特征，形状为[B, C, H, W]（与输入维度完全一致）
        """
        x = self.stack(x)
        # x = self.activation(x)
        return x

#通道注意力
class CAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CAM, self).__init__()
        self.F_squeeze_avg = nn.AdaptiveAvgPool2d(1)
        self.F_squeeze_max = nn.AdaptiveMaxPool2d(1)

        self.F_excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            # Sigmoid outside compare to SE
        )
    def forward(self, input):
        B,  C,  H,  W = input.size()
        y_avg = self.F_squeeze_avg( input ).view( B, C )
        y_max = self.F_squeeze_max( input ).view( B, C )

        y_merge = self.F_excitation( y_avg ).view( B, C, 1, 1) + self.F_excitation( y_max ).view( B, C, 1, 1)
        output = F.sigmoid(y_merge) * input
        return output


class MultiKernelDiff(nn.Module):
    def __init__(self, in_channels):
        super(MultiKernelDiff, self).__init__()
        # 实例化6个不同卷积核的卷积层，大小分别为3,5,5,7,7,9
        self.conv3 = MuitlConv(kernel_size=3, in_channels=in_channels)
        self.conv5_1 = MuitlConv(kernel_size=5, in_channels=in_channels)
        self.conv5_2 = MuitlConv(kernel_size=5, in_channels=in_channels)
        self.conv7_1 = MuitlConv(kernel_size=7, in_channels=in_channels)
        self.conv7_2 = MuitlConv(kernel_size=7, in_channels=in_channels)
        self.conv9 = MuitlConv(kernel_size=9, in_channels=in_channels)

        self.cam = CAM(channel=in_channels*3)

    def forward(self, x, y):
        # 分别通过6个卷积层
        out3 = self.conv3(x)
        out5_1 = self.conv5_1(x)
        out7_1 = self.conv7_1(x)

        out5_2 = self.conv5_2(y)
        out7_2 = self.conv7_2(y)
        out9 = self.conv9(y)
        
        # 计算差值：3-5、5-7、7-9
        diff1 = out3 - out5_2  # 卷积核3的输出 - 第2个卷积核5的输出
        diff2 = out5_1 - out7_2  # 第1个卷积核5的输出 - 第2个卷积核7的输出
        diff3 = out7_1 - out9  # 第1个卷积核7的输出 - 卷积核9的输出
        
        # 沿通道维度拼接（dim=1）
        result = torch.cat([diff1, diff2, diff3], dim=1)
        result = self.cam(result)
        return result




class Conv1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0, stride=1, dilation=1, groups=1):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
    def forward(self,x):
        return self.conv(x)
class Sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(Sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x
class ConvLeakyRelu2d(nn.Module):
    # convolution
    # leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups)
        # self.bn   = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        # print(x.size())
        return F.leaky_relu(self.conv(x), negative_slope=0.2)
class DenseBlock(nn.Module):
    def __init__(self,channels):
        super(DenseBlock, self).__init__()
        self.conv1 = ConvLeakyRelu2d(channels, channels)
        self.conv2 = ConvLeakyRelu2d(2*channels, channels)
        # self.conv3 = ConvLeakyRelu2d(3*channels, channels)
    def forward(self,x):
        x=torch.cat((x,self.conv1(x)),dim=1)
        x = torch.cat((x, self.conv2(x)), dim=1)
        return x
#计算亮度的边缘模块
class RGBD(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(RGBD, self).__init__()
        self.dense =DenseBlock(in_channels)
        self.convdown=Conv1(3*in_channels,out_channels)
        self.sobelconv=Sobelxy(in_channels)
        self.convup =Conv1(in_channels,out_channels)
    def forward(self,x):
        x1=self.dense(x)
        x1=self.convdown(x1)
        #visualize_compressed_feature(x1,1,"/root/shared-nvme/zpc/SOCODE/tempdata/Sobel_x1W.png")
        x2=self.sobelconv(x)
        x2=self.convup(x2)
        #visualize_compressed_feature(x2,1,"/root/shared-nvme/zpc/SOCODE/tempdata/Sobel_x2E.png")
        return F.leaky_relu(x1+x2,negative_slope=0.1) #+x2
#直方图提取
class color_extra(nn.Module):
    def __init__(self, channels):
        super(color_extra, self).__init__()

        self.block1 = nn.Sequential(nn.Conv2d(channels, 256, kernel_size=1, stride=1),
                                    nn.AdaptiveAvgPool2d((channels, 1)),
                                    nn.Softmax(dim=2))
        self.linear = nn.Linear(256, channels)
        self.relu = nn.LeakyReLU()

    def forward(self, raw, gt=None):

        # b, c, h, w = raw.shape
        hist = self.relu(self.linear(self.block1(raw).permute(0, 3, 2, 1).squeeze(1)))
        if gt is not None:
            gt_hist = self.relu(self.linear(self.block1(raw).permute(0, 3, 2, 1).squeeze(1)))
        else:
            gt_hist =None
        return hist, gt_hist

#修改部分
class MGCC(nn.Module):
    def __init__(self):
    # def __init__(self):
        super(MGCC, self).__init__()
        # self.dwt = DWT()
        # self.iwt = IWT()
        # 频域
        # self.before_unet = Before_Unet(wavelet_type='haar')
        # self.high_Encoder = High_Encoder()
        self.low_Encoder = Low_Encoder()
        # self.hlMerge_Deep_Features = HLMerge_Deep_Features(512,512)
        # self.hlMerge_Deep_Features = UNetConvBlock(512, 512)
        
        # self.conv = UNetConvBlock(512, 512)
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2)

        self.up1 = nn.Sequential(
            # 192→128：先两层3x3卷积提取空间特征，再1x1卷积降维
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),  # 1x1卷积降维至128通道
            LeakyReLU(),
            )
        
        self.conv_up1 = UNetConvBlock(512, 256)
        
        # self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up2 = nn.Sequential(
            # 192→128：先两层3x3卷积提取空间特征，再1x1卷积降维
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0),  # 1x1卷积降维至128通道
            LeakyReLU(),
            )
        self.conv_up2 = UNetConvBlock(256, 128)
        
        # self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up3 = nn.Sequential(
            # 192→128：先两层3x3卷积提取空间特征，再1x1卷积降维
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),  # 1x1卷积降维至128通道
            LeakyReLU(),
            )
        # self.conv_up3 = UNetConvBlock(192, 64)
        
        # self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        
        # self.convlast1 = UNetConvBlock(192,64)

        self.convlast1 = nn.Sequential(
            # 128→64：同理，两层3x3 + 1x1降维
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),

            # 64→32
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),

            # 32→16
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),

            # 16→4：最终降维至4通道
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            LeakyReLU(),
            nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0),
            LeakyReLU()
        )
        # self.convFirstlast = UNetConvBlock(4,4)
        #进入二阶段
        # self.second_stagebefore = UNetConvBlock(4,64)
        self.second_stagebefore = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3,padding=1),
            LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3,padding=1),
            LeakyReLU(),
            nn.Conv2d(32, 64, kernel_size=3,padding=1),
            LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            LeakyReLU()
        )

        #用于把G通道数归一化到和RB一致
        # self.Convg = UNetConvBlock(32,16)
        self.Convg = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),  # 1x1卷积降维，尺寸不变
            LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=1, stride=1, padding=0),
            LeakyReLU()
        )

        #多尺度对立色
        self.MSO_RG = MultiKernelDiff(16)
        self.MSO_YB = MultiKernelDiff(16)

        self.g_conv_RG = nn.Sequential(nn.Conv2d(96, 96, 3, 1, 1), LeakyReLU(),
                                          nn.Conv2d(96, 64, 3, 1, 1), LeakyReLU(),
                                          nn.Conv2d(64, 64, 3, 1, 1),LeakyReLU(),
                                          nn.Conv2d(64, 32, 3, 1, 1),LeakyReLU(),
                                          nn.Conv2d(32, 16, 3, 1, 1),LeakyReLU()
        )

        self.g_conv_YB = nn.Sequential(nn.Conv2d(96, 96, 3, 1, 1), LeakyReLU(),
                                          nn.Conv2d(96, 64, 3, 1, 1), LeakyReLU(),
                                          nn.Conv2d(64, 64, 3, 1, 1),LeakyReLU(),
                                          nn.Conv2d(64, 32, 3, 1, 1),LeakyReLU(),
                                          nn.Conv2d(32, 16, 3, 1, 1),LeakyReLU()
        )
        # self.conv_residual = nn.Conv2d(32, 128, kernel_size=1)  # 1x1卷积仅调整通道数，不改变空间维度
        self.conv_residual = nn.Sequential(nn.Conv2d(128, 64, 1), LeakyReLU(),
                                          nn.Conv2d(64, 32, 1), LeakyReLU()
        )

        self.w_r = nn.Parameter(torch.tensor(0.299, dtype=torch.float32))  # r 通道权重
        self.w_g = nn.Parameter(torch.tensor(0.587, dtype=torch.float32))  # g 通道权重
        self.w_b = nn.Parameter(torch.tensor(0.114, dtype=torch.float32))  # b 通道权重
        # self.guFilter1_5 = GuFilter(sigma=1.5,channels=16)
        #边缘计算算子
        self.sobel = RGBD(in_channels=16,out_channels=16)
        self.SA = Spatial_Attention()
        self.patch_embed_C = OverlapPatchEmbed(patch_size=3, stride=1, in_chans=32, embed_dim=64)#B,H,W,C
        self.patch_embed_S = OverlapPatchEmbed(patch_size=3, stride=1, in_chans=16, embed_dim=64)#B,H,W,C
        self.fuse = CrossAttentionBlock(64,64,64)
        #图像恢复
        # self.g_conv_post2 = nn.Sequential(nn.Conv2d(48, 48, 3, 1, 1),LeakyReLU(),
        #                                   nn.Conv2d(32, 16, 3, 1, 1),LeakyReLU(),
        #                                   nn.Conv2d(16, 12, 3, 1, 1),LeakyReLU()                    
        # )
        self.g_conv_post2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            LeakyReLU(),
            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            LeakyReLU(),
            nn.Conv2d(16, 12, kernel_size=1, stride=1, padding=0),
            LeakyReLU(),
            )

        self.up = nn.PixelShuffle(2)
        self.finish_second = UNetConvBlock(3,3) 
        
    def forward(self,x):
        # space = self.conv_space(x)
        # print(f"space图像的大小为{space.shape},而它的类型为{type(space)}") # space图像的大小为torch.Size([1, 16, 512, 512]),而它的类型为<class 'torch.Tensor'>
        # L,H = self.before_unet(x)
        # original = x
        # L,H = self.dwt(x)
        # print(f"低频图像的大小为{L.shape},而它的类型为{type(L)}") # 低频图像的大小为torch.Size([1, 16, 256, 256]),而它的类型为<class 'torch.Tensor'>
        # print(f"高频图像的大小为{H.shape},而它的类型为{type(H)}") # 高频图像的大小为torch.Size([1, 48, 256, 256]),而它的类型为<class 'torch.Tensor'>
        # H1,H2,H3,H4 = self.high_Encoder(H)
        L1,L2,L3,L4 = self.low_Encoder(x)  #L1 6,64,512,512. L2 6,128,256,256. L3 6,256,128,128. L4 6,256,64,64 
        # print(L1.shape)
        # x = torch.cat([L4,H4],dim=1) 
        # x = self.hlMerge_Deep_Features(x)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}") # x的大小为torch.Size([1, 512, 32, 32]),而x的类型为<class 'torch.Tensor'>
        x = self.up1(L4)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}") # x的大小为torch.Size([1, 256, 64, 64]),而x的类型为<class 'torch.Tensor'>
        x = torch.cat([x,L3],dim=1)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}") # x的大小为torch.Size([1, 768, 64, 64]),而x的类型为<class 'torch.Tensor'>
        # print(f"L4的大小为{L4.shape},而L4的类型为{type(L4)}") # x的大小为torch.Size([1, 768, 32, 32]),而x的类型为<class 'torch.Tensor'>
        # print(f"H4的大小为{H4.shape},而H4的类型为{type(H4)}") # x的大小为torch.Size([1, 768, 32, 32]),而x的类型为<class 'torch.Tensor'>
        x = self.conv_up1(x)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}") # x的大小为torch.Size([1, 256, 64, 64]),而x的类型为<class 'torch.Tensor'>
        
        x = self.up2(x)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}") # x的大小为torch.Size([1, 128, 128, 128]),而x的类型为<class 'torch.Tensor'>
        x = torch.cat([x,L2],dim=1)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}") # x的大小为torch.Size([1, 384, 128, 128]),而x的类型为<class 'torch.Tensor'>
        x = self.conv_up2(x)
        
        x = self.up3(x)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}") # x的大小为torch.Size([1, 64, 256, 256]),而x的类型为<class 'torch.Tensor'>
        x = torch.cat([x,L1],dim=1)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}") # x的大小为torch.Size([1, 192, 256, 256]),而x的类型为<class 'torch.Tensor'>

        x_RAW = self.convlast1(x)
        # x = self.convlast2(x)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}") # x的大小为torch.Size([1, 16, 256, 256]),而x的类型为<class 'torch.Tensor'>
        # x_RAW = self.iwt(x)#如果中间通过loss监督，那可以通过这里输出一阶段变量

        # print(f"x的大小为{x.shape},而x的类型为{type(x)}")     # x的大小为torch.Size([1, 4, 512, 512]),而x的类型为<class 'torch.Tensor'>
        # long_raw = self.convFirstlast(x)
        # x = get_detail(x)
        x = self.second_stagebefore(x_RAW)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}")
        r,G1,G2,b = split_feature_map(x)
        # print(f"R的大小为{R.shape},而x的类型为{type(R)}") # R的大小为torch.Size([1, 16, 512, 512]),而R的类型为<class 'torch.Tensor'>
        # print(f"B的大小为{B.shape},而B的类型为{type(B)}") # R的大小为torch.Size([1, 16, 512, 512]),而R的类型为<class 'torch.Tensor'>
        g = torch.cat([G1,G2],dim=1)
        # print(f"g的大小为{g.shape},而g的类型为{type(g)}")  # g的大小为torch.Size([1, 32, 512, 512]),而g的类型为<class 'torch.Tensor'>
        g = self.Convg(g)
        # print(f"g的大小为{g.shape},而g的类型为{type(g)}")  # g的大小为torch.Size([1, 16, 512, 512]),而g的类型为<class 'torch.Tensor'>
        #visualize_compressed_feature(r,1,"/root/shared-nvme/zpc/SOCODE/tempdata/r.png")
        #visualize_compressed_feature(g,1,"/root/shared-nvme/zpc/SOCODE/tempdata/g.png")
        #visualize_compressed_feature(b,1,"/root/shared-nvme/zpc/SOCODE/tempdata/b.png")

        #这里分离出RGB三个通道，其维度都为[1, 16, 512, 512]，在这里增加多尺度对立色计算的代码

        MSO_RG = self.MSO_RG(r,g)
        MSO_GR = self.MSO_RG(g,r)
        #visualize_compressed_feature(MSO_RG,1,"/raid/hbj/JDD/tempdata/MSO_RG.png")
        #visualize_compressed_feature(-MSO_RG,1,"/raid/hbj/JDD/tempdata/MSO_GR.png")
        # MSO_GR= -MSO_RG
        MSO_BY = self.MSO_YB(b,(r+g)/2)
        MSO_YB = self.MSO_YB((r+g)/2,b)
        #visualize_compressed_feature(MSO_BY,1,"/raid/hbj/JDD/tempdata/MSO_BY.png")
        #visualize_compressed_feature(-MSO_BY,1,"/raid/hbj/JDD/tempdata/MSO_YB.png")
        # MSO_YB= -MSO_BY
        
        #对立色彩信息
        Color_RG = self.g_conv_RG(torch.cat([MSO_RG, MSO_GR],dim=1)) #C=16  B,16,512,512
        Color_YB = self.g_conv_YB(torch.cat([MSO_BY, MSO_YB],dim=1))
        #visualize_compressed_feature(Color_RG,1,"/root/shared-nvme/zpc/SOCODE/tempdata/Color_RG.png")
        #visualize_compressed_feature(Color_YB,1,"/root/shared-nvme/zpc/SOCODE/tempdata/Color_YB.png")
        #visualize_compressed_feature(torch.cat([Color_RG,Color_YB],dim=1),1,"/root/shared-nvme/zpc/SOCODE/tempdata/Color.png")

        # W = (self.guFilter1_5(r)+self.guFilter1_5(b)+self.guFilter1_5(g))/math.sqrt(3)   #C=16  B,16,512,512
        # W = 0.299*self.guFilter1_5(r)+ 0.587*self.guFilter1_5(g)+0.114*self.guFilter1_5(b)
        W = self.w_r * r + self.w_g * g + self.w_b * b
        #visualize_compressed_feature(W,1,"/root/shared-nvme/zpc/SOCODE/tempdata/W.png")
        #亮度，边缘，色彩协同优化
        Sobel_L = self.sobel(W) #C=16  B,16,512,512
        #visualize_compressed_feature(Sobel_L,1,"/root/shared-nvme/zpc/SOCODE/tempdata/Sobel_L.png")
        concat_color = torch.cat([Color_RG, Color_YB],dim=1)

        Color_e = self.patch_embed_C(concat_color)#B,512,512,64
        Sobel_L = self.patch_embed_S(Sobel_L)#B,512,512,64
        Color_e = self.fuse(Sobel_L,Color_e)#自注意力优化远程色彩信息 B,128,512,512

        #残差拼接
        Color_residual = self.conv_residual(Color_e)
        Color_e = concat_color+Color_residual

        #visualize_compressed_feature(Color_e,1,"/root/shared-nvme/zpc/SOCODE/tempdata/Color_e.png")
        W = self.SA(W)
        # visualize_compressed_feature(Color_e,1,"/root/shared-nvme/zpc/SOCODE/tempdata/SA_W.png")
        Color_e = Color_e*W
        # x = self.g_conv_post2(torch.cat([W,Color_e],dim=1))
        x = self.g_conv_post2(Color_e)
        #visualize_compressed_feature(x,1,"/root/shared-nvme/zpc/SOCODE/tempdata/x_post2.png")
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}")     # x的大小为torch.Size([1, 12, 512, 512]),而x的类型为<class 'torch.Tensor'>
        x = self.up(x)
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}")     # x的大小为torch.Size([1, 3, 1024, 1024]),而x的类型为<class 'torch.Tensor'>
        x = self.finish_second(x)
        #visualize_compressed_feature(x,3,"/root/shared-nvme/zpc/SOCODE/tempdata/x_finish.png")
        # print(f"x的大小为{x.shape},而x的类型为{type(x)}")     # x的大小为torch.Size([1, 3, 1024, 1024]),而x的类型为<class 'torch.Tensor'>
        return x,x_RAW

if __name__ == '__main__':
    
    # 创建了一个大小为 (1, 3, 1024, 1024) 的 NumPy 数组，表示输入数据。这是一个包含 1 个样本、3 个通道、大小为 1024 x 1024 的图像
    input = np.ones((1,16,512,512),dtype=np.float32)
    
    # 这里将 NumPy 数组转换为 PyTorch 张量，并将其移动到 CUDA 设备（GPU）上
    input = torch.tensor(input, dtype=torch.float32, device='cuda')

    
    # print(f"input的大小为{input.shape},而它的类型为{type(input)}") # input的大小为torch.Size([1, 4, 512, 512]),而它的类型为<class 'torch.Tensor'>
    print(f"input的大小为{input.shape},而它的类型为{type(input)}")
    # print(input)
   
    net = GuFilter()
    # 将模型移动到 CUDA 设备（GPU）上，如果可用的话。这意味着后续的计算将在 GPU 上执行
    net.cuda()
    with torch.no_grad():
        gt = net(input)
    # torch.Size([1, 3, 1024, 1024])
    # print(f"long_raw的大小为{long_raw.shape},而它的类型为{type(long_raw)}")
    print(f"gt的大小为{gt.shape},而它的类型为{type(gt)}")
    # print(gt[1])
    