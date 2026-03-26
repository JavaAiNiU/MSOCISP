import argparse
from ast import arg
import logging
import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  
import numpy as np
from models.model_MSO_CCA_RAWLossNoduili import *
from datasets.ZRRMCRDataset import *
from utils.metrics import get_psnr_torch, get_ssim_torch, get_lpips_torch

from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from scipy.interpolate import griddata
plt.switch_backend('agg')



# # ================= 1. 定义三个不同的权重提取器 =================

# def get_kernel_layer1(conv_module):
#     """提取第一层卷积 (stack[0]) 的空间平均权重"""
#     weight_tensor = conv_module.stack[0].weight.data.cpu()
#     return weight_tensor.mean(dim=(0, 1)).numpy()

# def get_kernel_layer2(conv_module):
#     """提取第二层卷积 (stack[2]) 的空间平均权重"""
#     weight_tensor = conv_module.stack[2].weight.data.cpu()
#     return weight_tensor.mean(dim=(0, 1)).numpy()

# def get_kernel_sum(conv_module):
#     """提取第一层和第二层卷积之和的空间平均权重"""
#     w1 = conv_module.stack[0].weight.data.cpu()
#     w2 = conv_module.stack[2].weight.data.cpu()
#     weight_tensor = w1 + w2
#     return weight_tensor.mean(dim=(0, 1)).numpy()

# # ================= 2. 绘图主函数 =================
# def save_clean_smoothed_surfaceRG(conv_small, conv_large, name, extractor_func, save_dir="multi_layer_kernels_with_bg"):
#     """
#     生成带有默认3D背景板、但无XYZ坐标轴标识、无数值刻度的平滑双层曲面图。
#     """
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
        
#     k_small_raw = extractor_func(conv_small)
#     k_large_raw = extractor_func(conv_large)
    
#     # 基线对齐
#     k_small_shifted = k_small_raw - np.min(k_small_raw)
#     k_large_neg = -k_large_raw
#     k_large_neg_shifted = k_large_neg - np.max(k_large_neg)
    
#     size_small = k_small_raw.shape[0]
#     size_large = k_large_raw.shape[0]
    
#     # 立方插值平滑数据
#     N_hires = 100 
#     half_l = size_large / 2.0
#     xi = np.linspace(-half_l, half_l, N_hires)
#     yi = np.linspace(-half_l, half_l, N_hires)
#     X_hires, Y_hires = np.meshgrid(xi, yi)

#     def interp_kernel(kernel, source_size, target_X, target_Y):
#         half_s = source_size / 2.0
#         x_sparse = np.linspace(-(source_size // 2), source_size // 2, source_size)
#         X_sparse, Y_sparse = np.meshgrid(x_sparse, x_sparse)
#         points = np.vstack((X_sparse.flatten(), Y_sparse.flatten())).T
#         values = kernel.flatten()
#         grid_z = griddata(points, values, (target_X, target_Y), method='cubic', fill_value=0)
#         return grid_z

#     k_small_smooth = interp_kernel(k_small_shifted, size_small, X_hires, Y_hires)
#     k_large_neg_smooth = interp_kernel(k_large_neg_shifted, size_large, X_hires, Y_hires)

#     # ================= 绘图与美化 =================
#     fig = plt.figure(figsize=(10, 8))
#     ax = fig.add_subplot(111, projection='3d')

#     # 绘制曲面
#     ax.plot_surface(X_hires, Y_hires, k_large_neg_smooth, cmap='Greens_r', 
#                     alpha=0.7, edgecolor='none', rstride=1, cstride=1, antialiased=True)
#     ax.plot_surface(X_hires, Y_hires, k_small_smooth, cmap='Reds', 
#                     alpha=0.8, edgecolor='none', rstride=1, cstride=1, antialiased=True)
    
#     # 保留网格与背景板 (Matplotlib 默认行为，确保不被隐藏)
#     ax.grid(True)

#     # 隐藏坐标轴上的数值刻度 (只保留线，不要数字)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_zticks([])

#     # 【核心修改】：彻底不设置 set_xlabel, set_ylabel, set_zlabel，从而去除 XYZ 标识
    
#     # 调整视角
#     ax.view_init(elev=20, azim=45) 
#     z_max = np.max(k_small_smooth)
#     z_min = np.min(k_large_neg_smooth)
#     ax.set_zlim(z_min * 1.1, z_max * 1.1)

#     # 保存静态高清 PNG (去除了 transparent=True，背景将带有正常的白色底和灰色3D板)
#     png_path = os.path.join(save_dir, f"{name.replace(' ', '_')}.png")
#     plt.savefig(png_path, dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # ================= 交互式 HTML 同步美化 =================
#     try:
#         import plotly.graph_objects as go
#         fig_html = go.Figure()
        
#         fig_html.add_trace(go.Surface(z=k_large_neg_smooth, x=X_hires, y=Y_hires, colorscale='Greens_r', showscale=False, opacity=0.8))
#         fig_html.add_trace(go.Surface(z=k_small_smooth, x=X_hires, y=Y_hires, colorscale='Reds', showscale=False, opacity=0.9))
        
#         # Plotly 中恢复网格(showgrid=True)，同时隐藏刻度数值和标题标识(title='')
#         clean_axis = dict(showgrid=True, zeroline=False, showticklabels=False, showaxeslabels=False)
        
#         fig_html.update_layout(
#             title=f'{name}', 
#             autosize=False, width=800, height=800,
#             scene=dict(
#                 xaxis=dict(**clean_axis, title=''),
#                 yaxis=dict(**clean_axis, title=''),
#                 zaxis=dict(**clean_axis, title='', range=[z_min * 1.1, z_max * 1.1])
#             )
#             # 移除了 paper_bgcolor 和 plot_bgcolor 设置，恢复默认背景
#         )
        
#         html_path = os.path.join(save_dir, f"{name.replace(' ', '_')}.html")
#         fig_html.write_html(html_path)
#         print(f"[{name}] 成功保存 (带背景, 无XYZ标识) PNG 和 HTML 至 '{save_dir}' 文件夹！")
        
#     except ImportError:
#         pass

# from PIL import Image
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6  # 转换为百万（M）

def lpips_norm(img, device='cuda'):
	img = img[:, :, :, np.newaxis].transpose((3, 2, 0, 1))
	img = img / (255. / 2.) - 1
	return torch.Tensor(img).to(device)
def calc_lpips(out, target, loss_fn_alex):
	lpips_out = lpips_norm(out)
	lpips_target = lpips_norm(target)
	LPIPS = loss_fn_alex(lpips_out, lpips_target)
	return LPIPS.detach().cpu().item()

def test(args):
    # device
    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    # 如果'logs'文件夹不存在，则创建一个
    logs_folder =  os.path.join(args.result_dir, 'logs')
    images_folder =  os.path.join(args.result_dir, 'images')
    
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    
    
    # 设置日志记录
    logging.basicConfig(filename=os.path.join(logs_folder, 'test.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

    _,test_loader = LLdataset(dir=args.data_dir,patch_size=256,batch_size=args.batch_size,train_file=args.train_list_file, 
                                         test_file=args.test_list_file,num_workers=2).get_loaders()

    
    # model
    # model = SIDUNet()
    model = MGCC().to(device)

    # GPU数量大于1时，使用四张卡并行计算
    if torch.cuda.device_count() > 1:
        # 确保可用GPU数量不少于4张
        num_available_gpus = torch.cuda.device_count()
        if num_available_gpus >= 4:
            use_devices = [0, 1, 2, 3]  # 指定使用前4张GPU
        else:
            # 如果可用GPU不足4张，使用所有可用GPU
            use_devices = list(range(num_available_gpus))
            print(f"可用GPU数量不足4张，将使用{num_available_gpus}张GPU")
        
        print(f"Let's use {len(use_devices)} GPUs: {use_devices}!")
        # 使用DataParallel包装模型，指定要使用的GPU列表
        model = nn.DataParallel(model, device_ids=use_devices)
        model.to(device)  # device通常为"cuda"（默认使用第一张卡作为主卡）
    model.to(device)
    state_dict = torch.load(args.model)
    # 2. 去除所有键的"module."前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        # 截取键中"module."之后的部分（如果存在）
        key = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[key] = v
    # 3. 加载处理后的权重
    model.load_state_dict(new_state_dict) 
    # model.load_state_dict(torch.load(args.model))
    total_params = count_parameters(model)
    print(f'Total trainable parameters: {total_params}')
    model.eval()


    # 获取迭代器的长度
    total_batches = len(test_loader)
    print(total_batches) # 598
    # 初始化 PSNR 和 SSIM 列表
    psnr_list = []
    ssim_list = []
    lpips_list = []
    # loss_fn_alex_v1 = lpips.LPIPS(net='alex', version='0.1').to(device)
    with torch.no_grad(): 
        # testing
        agvtime=0
        count =0
        for i, databatch in tqdm(enumerate(test_loader), total=total_batches):
            etime = time.time()
            # input_path, gt_path = data['low_RAW_img'][0], data['high_RGB_img'][0]#, data['ratio'][0]
            # index = data['index']
            input_raw = databatch['input_raw'].cuda(non_blocking=True)
            # gt_raw = databatch['gt_raw'].cuda(non_blocking=True)
            gt_rgb = databatch['gt_rgb'].cuda(non_blocking=True)
            input_path = databatch['index'][0]




            signal_time = time.time()
            pred_rgb,_ = model(input_raw)#pred_rgb, pred_raw
            agvtime=agvtime+(time.time()-signal_time)
            count+=1
            # print(time.time() - etime)

            pred_rgb = torch.clamp(pred_rgb, 0, 1)
            gt_rgb = torch.clamp(gt_rgb, 0, 1)
            lpips = get_lpips_torch(pred_rgb, gt_rgb)

            preds = torch.clip(pred_rgb * 255.0, 0, 255.0)
            gt_rgb = torch.clip(gt_rgb * 255.0, 0, 255.0)
            
            single_psnr = get_psnr_torch(preds, gt_rgb)
            single_ssim = get_ssim_torch(preds, gt_rgb)

            # 将 PSNR 和 SSIM 添加到列表中
            psnr_list.append(single_psnr.mean().item())
            ssim_list.append(single_ssim.mean().item())
            lpips_list.append(lpips.item())
            logging.info(f"PSNR: {single_psnr.mean().item()},SSIM: {single_ssim.mean().item()},LPIPS: {lpips.item()},SIGTime: {signal_time - etime},Time: {time.time() - etime}")


            pred_rgb =  preds.round()
            pred_rgb = pred_rgb.squeeze(0).cpu().numpy().astype(np.uint8)
            save_path = os.path.join(images_folder, os.path.basename(input_path)) + '.png'
            Image.fromarray(pred_rgb.transpose(1, 2, 0)).save(save_path)

            gt_rgb = gt_rgb.round()
            gt_rgb = gt_rgb.squeeze(0).cpu().numpy().astype(np.uint8)
            save_path = os.path.join(images_folder, os.path.basename(input_path)) + '_gt.png'
            Image.fromarray(gt_rgb.transpose(1, 2, 0)).save(save_path)

            
        # 计算平均 PSNR 和 SSIM
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_lpips = np.mean(lpips_list) 
        logging.info(f"Average PSNR: {avg_psnr},Average SSIM: {avg_ssim},Average LPIPS: {avg_lpips},Average time :{agvtime/count}")

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    parser = argparse.ArgumentParser(description="evaluating model")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--data_dir', type=str, default='/root/shared-nvme/MAI')
    parser.add_argument('--train_list_file', type=str, default='/root/shared-nvme/MAI/data/ISP/AAAI-25/MAI_dataset/train.txt')
    parser.add_argument('--test_list_file',  type=str, default='/root/shared-nvme/MAI/data/ISP/AAAI-25/MAI_dataset/val.txt')

    parser.add_argument('--result_dir', type=str, default='/root/shared-nvme/MSOCO/test_result/test_MAINoduili/best_psnr_model')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1, help='multi-threads for data loading')
    parser.add_argument('--model', type=str, default='')# pretrain model path
    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    test(args)