import argparse
from ast import arg
import os
from telnetlib import PRAGMA_HEARTBEAT
import time
import torch
from torch import optim
import torch.nn.functional as F

from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torch.utils.data import DataLoader
from utils.metrics import get_psnr_torch, get_ssim_torch, get_lpips_torch
from utils.util import *
from models.model_MSO_CCA_RAWLoss import *
from Losses.wavelet_loss import CombinedLoss
from Losses.color_loss import ColorHistogramKLLoss
from Losses.vgg_loss import VGGPerceptualLoss  # 新增：感知损失导入
from datasets.ZRRMCRDataset import *
from PIL import Image
from tqdm import tqdm

#混合精度
from torch.cuda.amp import autocast, GradScaler

# 色彩一致性损失
def ccl_loss(input_img, gt_img):
    input_img, gt_img = torch.clip(input_img * 255, 0, 255), \
        torch.clip(gt_img * 255, 0, 255)
    obj = ColorHistogramKLLoss()
    loss = obj(input_img, gt_img).abs()
    return loss.cuda()

def count_model_params(model, unit="M"):
    """
    计算模型总参数量，并转换为指定单位（默认M）
    :param model: PyTorch模型
    :param unit: 单位，支持 'M'（百万）、'K'（千）、'B'（十亿）
    :return: 参数量（浮点数）
    """
    total_params = sum(p.numel() for p in model.parameters())  # 累加所有参数的元素数量
    if unit == "M":
        return total_params / 1e6  # 转换为百万（M）
    elif unit == "K":
        return total_params / 1e3  # 转换为千（K）
    elif unit == "B":
        return total_params / 1e9  # 转换为十亿（B）
    else:
        return total_params  # 不转换，返回原始数量
    
def train_and_evaluate(args, alpha, beta):
    # 创建权重组合子文件夹
    weight_dir = os.path.join(args.result_dir, f'alpha_{alpha}_beta_{beta}')
    if not os.path.exists(weight_dir):
        os.makedirs(weight_dir)
    
    save_bestmodel = os.path.join(weight_dir, 'best_loss_model')
    if not os.path.exists(save_bestmodel):
        os.makedirs(save_bestmodel)
    save_lastmodel = os.path.join(weight_dir, 'last_model')
    if not os.path.exists(save_lastmodel):
        os.makedirs(save_lastmodel)
    
    save_best_psnr_model = os.path.join(weight_dir, 'best_psnr_model')
    if not os.path.exists(save_best_psnr_model):
        os.makedirs(save_best_psnr_model)
    save_best_ssim_model = os.path.join(weight_dir, 'best_ssim_model')
    if not os.path.exists(save_best_ssim_model):
        os.makedirs(save_best_ssim_model)
    logs_folder = os.path.join(weight_dir, 'logs') 
    if not os.path.exists(logs_folder):
        os.makedirs(logs_folder)
    save_best_val_loss_model = os.path.join(weight_dir, 'best_val_loss_model')
    if not os.path.exists(save_best_val_loss_model):
        os.makedirs(save_best_val_loss_model)
    
    save_best_val_psnr_model = os.path.join(weight_dir, 'best_val_psnr_model')
    if not os.path.exists(save_best_val_psnr_model):
        os.makedirs(save_best_val_psnr_model)
    
    save_best_val_ssim_model = os.path.join(weight_dir, 'best_val_ssim_model')
    if not os.path.exists(save_best_val_ssim_model):
        os.makedirs(save_best_val_ssim_model)

    # 数据集加载：完全保留原文件配置，未做任何修改
    train_loader,test_loader = LLdataset(dir=args.data_dir,patch_size=256,batch_size=args.batch_size,train_file=args.train_list_file, 
                                         test_file=args.test_list_file,num_workers=4).get_loaders()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MGCC().to(device)
    
    # 新增：计算并打印模型参数量
    param_count = count_model_params(model)
    print(f"模型总参数量: {param_count:.2f} M")  # 保留2位小数

    # 损失函数初始化：新增VGG感知损失criterion_vgg
    criterion_wave = CombinedLoss().cuda()
    criterion = torch.nn.L1Loss().cuda()
    color_loss = ColorHistogramKLLoss().cuda()
    criterion_vgg = VGGPerceptualLoss().cuda()  # 核心新增：感知损失
    
    # GPU数量大于1时，使用四张卡并行计算：完全保留原文件逻辑
    if torch.cuda.device_count() > 1:
        num_available_gpus = torch.cuda.device_count()
        if num_available_gpus >= 4:
            use_devices = [0, 1, 2, 3]
        else:
            use_devices = list(range(num_available_gpus))
            print(f"可用GPU数量不足4张，将使用{num_available_gpus}张GPU")
        
        print(f"Let's use {len(use_devices)} GPUs: {use_devices}!")
        model = nn.DataParallel(model, device_ids=use_devices)
        model.to(device)
   
    # 断点续训：完全保留原文件逻辑
    if args.resume:
        last_info_list = process_files(save_lastmodel)[0]
        model_name = last_info_list['文件名']
        model_loss = float(last_info_list['Loss'])
        model_epoch = int(last_info_list['epoch'])
        model_path = os.path.join(save_lastmodel, model_name)   
        state_dict = torch.load(model_path)
        new_state_dict = {}
        for k, v in state_dict.items():
            key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[key] = v
        model.load_state_dict(new_state_dict)
        model = model.to(device)
        lastepoch = model_epoch + 1
        print(f"从{lastepoch}轮继续训练")
        best_info_list = process_files(save_bestmodel)[0]
        min_loss = float(best_info_list['Loss'])
        psnr_info_list = process_files(save_best_psnr_model)[0]
        max_psnr = float(psnr_info_list['Loss'])
        ssim_info_list = process_files(save_best_ssim_model)[0]
        max_ssim = float(ssim_info_list['Loss'])
        try:
            val_loss_info = process_files(save_best_val_loss_model)[0]
            val_min_loss = float(val_loss_info['Loss'])
        except:
            val_min_loss = float('inf')
        try:
            val_psnr_info = process_files(save_best_val_psnr_model)[0]
            val_max_psnr = float(val_psnr_info['Loss'])
        except:
            val_max_psnr = 0
        try:
            val_ssim_info = process_files(save_best_val_ssim_model)[0]
            val_max_ssim = float(val_ssim_info['Loss'])
        except:
            val_max_ssim = 0
    else:
        lastepoch = 1
        min_loss = float('inf')
        max_psnr = 0
        max_ssim = 0
        val_min_loss = float('inf')
        val_max_psnr = 0
        val_max_ssim = 0
    print(f"lastepoch为 {lastepoch}")


    # 优化器：完全保留原文件配置
    G_opt = optim.Adam(model.parameters(), lr=args.lr)
    # 学习率调度器：保留原逻辑，注释verbose=True避免低版本PyTorch报错
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        G_opt,
        mode='max',          
        factor=0.8,          
        patience=10,         
        threshold=1e-2,      
        threshold_mode='abs',
        cooldown=10,          
        min_lr=1e-8,         
        # verbose=True  # 注释：避免低版本PyTorch报未知参数错误
    )
    # scaler = GradScaler()
    scaler = torch.amp.GradScaler('cuda')

    # 主训练循环
    for epoch in range(lastepoch,args.num_epoch):
        model.train()
        if os.path.isdir(weight_dir + '%04d' % epoch):
            continue
        etime = time.time()
        eloss = 0
        epsnr = 0
        essim = 0
        count = 0
        # 新增：初始化各损失分项累加变量，用于计算epoch平均损失
        total_L1 = 0
        total_wave = 0
        total_closs = 0
        total_vgg = 0

        total_batches = len(train_loader)
        # 新增：tqdm进度条-实时显示当前批次的总损失+各分项损失
        pbar = tqdm(enumerate(train_loader), total=total_batches,dynamic_ncols=True, ascii=True,
                    desc=f"Epoch {epoch:04d} [Loss: 0.000 | L1:0.000 | Wave:0.000 | Color:0.000 | VGG:0.000]")
        for i, databatch in pbar:
            input_raw = databatch["input_raw"].cuda(non_blocking=True)
            gt_rgb = databatch["gt_rgb"].cuda(non_blocking=True)
            count += 1

            # 模型推理：还原返回值，和损失计算统一
            # preds,x_raw = model(input_raw)

            # # 核心修改：计算各损失分项，新增感知损失loss_vgg
            # loss_wave = criterion_wave(preds, gt_rgb)
            # loss_L1 = criterion(preds, gt_rgb)
            # closs = color_loss(preds,gt_rgb)
            # loss_vgg = criterion_vgg(preds, gt_rgb)  # 新增：感知损失计算
            # loss = loss_L1 + loss_wave + 2*closs + loss_vgg  # 总损失加入感知损失

            # 反向传播：完全保留原逻辑
            G_opt.zero_grad()

            # 【核心修改】开启混合精度上下文
            with autocast():
                preds, x_raw = model(input_raw)
                
                # 所有的损失计算也在 FP16/FP32 混合下进行
                loss_wave = criterion_wave(preds, gt_rgb)
                loss_L1 = criterion(preds, gt_rgb)
                # closs = color_loss(preds, gt_rgb)
                closs = color_loss(preds.float(), gt_rgb.float())
                loss_vgg = criterion_vgg(preds, gt_rgb)
                
                loss = loss_L1 + loss_wave + 2 * closs + loss_vgg

            # 【核心修改】使用 scaler 进行反向传播和参数更新
            scaler.scale(loss).backward()
            scaler.step(G_opt)
            scaler.update()


            # loss.backward()
            # G_opt.step()

            # 【新增修复】：在这一步统一且仅调用一次 .item()，切断与 GPU 的联系
            l_val = loss.item()
            l1_val = loss_L1.item()
            wave_val = loss_wave.item()
            c_val = closs.item()
            vgg_val = loss_vgg.item()

            # 新增：累加各损失值（转item避免显存占用）
            eloss += l_val
            total_L1 += l1_val
            total_wave += wave_val
            total_closs += c_val
            total_vgg += vgg_val

            # PSNR/SSIM计算：完全保留原逻辑
            # preds = torch.clip(preds * 255.0, 0, 255.0)
            # gt_rgb = torch.clip(gt_rgb * 255.0, 0, 255.0)

            preds = torch.clip(preds.detach().float() * 255.0, 0, 255.0)
            gt_rgb = torch.clip(gt_rgb.detach().float() * 255.0, 0, 255.0)


            single_psnr = get_psnr_torch(preds, gt_rgb)
            single_ssim = get_ssim_torch(preds, gt_rgb)
            epsnr += single_psnr.mean().item()
            essim += single_ssim.mean().item()

            # 新增：实时更新tqdm进度条的损失显示（保留3位小数）
            pbar.set_description(
                            f"Epoch {epoch:04d} [Loss: {l_val:.3f} | L1:{l1_val:.3f} | Wave:{wave_val:.3f} | Color:{c_val:.3f} | VGG:{vgg_val:.3f}]"
                        )
        
        # 计算本轮epoch的平均损失和指标
        aloss = eloss/count
        apsnr = epsnr/count
        assim = essim/count
        avg_L1 = total_L1/count
        avg_wave = total_wave/count
        avg_closs = total_closs/count
        avg_vgg = total_vgg/count
         
        temp_loss = aloss
        temp_psnr = apsnr
        temp_ssim = assim

        # 新增：打印epoch平均损失，包含所有分项（保留4位小数，Color*2贴合计算逻辑）
        print(f"\nEpoch = {epoch}. \tTotal Loss = {aloss:.4f} | L1 = {avg_L1:.4f} | Wave = {avg_wave:.4f} | Color*2 = {2*avg_closs:.4f} | VGG = {avg_vgg:.4f}")
        print(f"\tPSNR = {apsnr:.4f}, \tSSIM = {assim:.4f},\tTime = {time.time() - etime:.2f}s")

        # 保存训练集最佳模型：保留原逻辑，优化文件名（损失保留4位小数）
        if temp_loss < min_loss:
            min_loss = temp_loss
            min_loss_epoch = epoch
            if os.path.exists(save_bestmodel):
                for filename in os.listdir(save_bestmodel):
                    if filename.endswith('.pth'):
                        os.remove(os.path.join(save_bestmodel, filename))
                bestmodel = os.path.join(save_bestmodel, "bestmodel_{:.4f}_{}.pth".format(temp_loss, epoch))
                torch.save(model.state_dict(), bestmodel)   
                  
        if temp_psnr > max_psnr:
            max_psnr = temp_psnr
            if os.path.exists(save_best_psnr_model):
                for filename in os.listdir(save_best_psnr_model):
                    if filename.endswith('.pth'):
                        os.remove(os.path.join(save_best_psnr_model, filename))
                bestpsnrmodel = os.path.join(save_best_psnr_model, "bestpsnrmodel_{:.4f}_{}.pth".format(temp_psnr, epoch))
                torch.save(model.state_dict(), bestpsnrmodel) 
        
        if temp_ssim > max_ssim:
            max_ssim = temp_ssim
            if os.path.exists(save_best_ssim_model):
                for filename in os.listdir(save_best_ssim_model):
                    if filename.endswith('.pth'):
                        os.remove(os.path.join(save_best_ssim_model, filename))
                bestssimmodel = os.path.join(save_best_ssim_model, "bestssimmodel_{:.4f}_{}.pth".format(temp_ssim, epoch))
                torch.save(model.state_dict(), bestssimmodel) 
        
        # 保存最后一轮模型：保留原逻辑，优化文件名
        if os.path.exists(save_lastmodel):
            for filename in os.listdir(save_lastmodel):
                if filename.endswith('.pth'):
                    os.remove(os.path.join(save_lastmodel, filename))
            lastmodel = os.path.join(save_lastmodel, "ModelSnapshot_{:.4f}_{}.pth".format(temp_loss, epoch))
            torch.save(model.state_dict(), lastmodel)

        # 验证阶段：保留原验证频率（每5epoch），核心修改-加入感知损失计算
        if epoch % 5 == 0:
            val_eloss = 0.0
            val_epsnr = 0.0
            val_essim = 0.0
            val_count = 0
            model.eval()
            with torch.no_grad():
                total_batches = len(test_loader)
                for i, databatch in tqdm(enumerate(test_loader), total=total_batches,dynamic_ncols=True,ascii=True, desc=f"Epoch {epoch} 验证"):
                    input_raw = databatch['input_raw'].cuda(non_blocking=True)
                    gt_rgb = databatch['gt_rgb'].cuda(non_blocking=True)
                    
                    val_count += 1
                    preds, x_raw = model(input_raw)

                    # 核心修改：验证集也加入感知损失，和训练集损失计算完全一致
                    loss_wave = criterion_wave(preds, gt_rgb)
                    loss_L1 = criterion(preds, gt_rgb)
                    closs = color_loss(preds, gt_rgb)
                    loss_vgg = criterion_vgg(preds, gt_rgb)  # 新增：验证集感知损失
                    loss = loss_L1 + loss_wave + 2 * closs + loss_vgg  # 验证集总损失

                    # 累加验证指标
                    val_eloss += loss.item()
                    preds = torch.clip(preds * 255.0, 0, 255.0)
                    gt_rgb = torch.clip(gt_rgb * 255.0, 0, 255.0)
                    val_epsnr += get_psnr_torch(preds, gt_rgb).mean().item()
                    val_essim += get_ssim_torch(preds, gt_rgb).mean().item()

            # 计算验证集平均指标
            val_loss = val_eloss / val_count
            val_psnr = val_epsnr / val_count
            val_ssim = val_essim / val_count
            scheduler.step(val_psnr)  

            # 打印验证结果：保留原逻辑
            print(f"\n===== Epoch {epoch} 验证结果 =====")
            print(f"验证样本数: {val_count}")
            print(f"验证集 - Loss: {val_loss:.4f}, PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}")
            print("================================\n")

            # 保存验证集最佳模型：保留原逻辑，优化文件名
            if val_loss < val_min_loss:
                val_min_loss = val_loss
                val_min_loss_epoch = epoch
                for filename in os.listdir(save_best_val_loss_model):
                    if filename.endswith('.pth'):
                        os.remove(os.path.join(save_best_val_loss_model, filename))
                best_val_loss_model = os.path.join(save_best_val_loss_model, f"bestvalloss_{val_loss:.4f}_{epoch}.pth")
                torch.save(model.state_dict(), best_val_loss_model)
                print(f"更新验证集最佳Loss模型（Epoch {epoch}，Loss: {val_loss:.4f}）")
            if val_psnr > val_max_psnr:
                val_max_psnr = val_psnr
                val_max_psnr_epoch = epoch
                for filename in os.listdir(save_best_val_psnr_model):
                    if filename.endswith('.pth'):
                        os.remove(os.path.join(save_best_val_psnr_model, filename))
                best_val_psnr_model = os.path.join(save_best_val_psnr_model, f"bestvalpsnr_{val_psnr:.4f}_{epoch}.pth")
                torch.save(model.state_dict(), best_val_psnr_model)
                print(f"更新验证集最佳PSNR模型（Epoch {epoch}，PSNR: {val_psnr:.4f}）")
            if val_ssim > val_max_ssim:
                val_max_ssim = val_ssim
                val_max_ssim_epoch = epoch
                for filename in os.listdir(save_best_val_ssim_model):
                    if filename.endswith('.pth'):
                        os.remove(os.path.join(save_best_val_ssim_model, filename))
                best_val_ssim_model = os.path.join(save_best_val_ssim_model, f"bestvalssim_{val_ssim:.4f}_{epoch}.pth")
                torch.save(model.state_dict(), best_val_ssim_model)
                print(f"更新验证集最佳SSIM模型（Epoch {epoch}，SSIM: {val_ssim:.4f}）")

            # 显存清理：新增删除loss_vgg，避免泄漏
            del val_eloss, val_epsnr, val_essim, val_count
            del preds, x_raw, loss_wave, loss_L1, closs, loss_vgg, loss
            del input_raw, gt_rgb
            torch.cuda.empty_cache()
        
    # 训练结束：打印最终结果，写入日志，关闭TensorBoard
    print(f"Min Loss: {min_loss:.4f} (Epoch {min_loss_epoch}), Max PSNR: {max_psnr:.4f},Max SSIM: {max_ssim:.4f} ") 

if __name__ == "__main__":
    # 完全保留原文件的所有参数配置、数据集路径，未做任何修改
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/root/shared-nvme/ZRR')
    parser.add_argument('--train_list_file', type=str, default='/root/shared-nvme/ZRR/train.txt')
    parser.add_argument('--test_list_file',  type=str, default='/root/shared-nvme/ZRR/test.txt')
    
    parser.add_argument('--train_datatype', type=str, default='train')
    parser.add_argument('--test_datatype',  type=str, default='test')
    parser.add_argument('--result_dir', type=str, default='/root/shared-nvme/MSOCO/Results/result_MSO_CCA/ZRR/L1_WaveLoss_sum_weights')
    parser.add_argument('--ps', type=int, default=512)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=30)#三卡72，双卡48
    parser.add_argument('--num_epoch', type=int, default=4001)
    parser.add_argument('--model_save_freq', type=int, default=200)
    parser.add_argument('--resume', type=bool, default=True, help='continue training')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    
    train_and_evaluate(args, 0.5, 0.5)