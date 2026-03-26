import argparse
import os
import time
import torch
from torch import optim
import torch.nn.functional as F
import scipy.io, scipy.misc
import scipy

from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torch.utils.data import DataLoader
from utils.metrics import get_psnr_torch, get_ssim_torch, get_lpips_torch
from utils.util import *
from models.model_MSO_CCA_RAWLoss import *
from Losses.wavelet_loss import CombinedLoss
from Losses.color_loss import ColorHistogramKLLoss
from Losses.vgg_loss import VGGPerceptualLoss  # 感知损失导入
from datasets.ZRRMCRDataset import *
from PIL import Image
from tqdm import tqdm

# 【新增】DDP 相关库
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

# 色彩一致性损失
def ccl_loss(input_img, gt_img):
    input_img, gt_img = torch.clip(input_img * 255, 0, 255), \
        torch.clip(gt_img * 255, 0, 255)
    obj = ColorHistogramKLLoss()
    loss = obj(input_img, gt_img).abs()
    return loss.to(input_img.device)  # 适配多卡 device

def count_model_params(model, unit="M"):
    total_params = sum(p.numel() for p in model.parameters())
    if unit == "M":
        return total_params / 1e6
    elif unit == "K":
        return total_params / 1e3
    elif unit == "B":
        return total_params / 1e9
    else:
        return total_params

# 【新增】分布式指标同步函数
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def train_and_evaluate(args, alpha, beta):
    # ================= 1. DDP 初始化 =================
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main_process = local_rank == 0  # 仅 rank 0 负责打印和保存
    
    # ================= 2. 目录创建与日志设置 (仅限主进程) =================
    weight_dir = os.path.join(args.result_dir, f'alpha_{alpha}_beta_{beta}')
    save_bestmodel = os.path.join(weight_dir, 'best_loss_model')
    save_lastmodel = os.path.join(weight_dir, 'last_model')
    save_best_psnr_model = os.path.join(weight_dir, 'best_psnr_model')
    save_best_ssim_model = os.path.join(weight_dir, 'best_ssim_model')
    logs_folder = os.path.join(weight_dir, 'logs') 
    save_best_val_loss_model = os.path.join(weight_dir, 'best_val_loss_model')
    save_best_val_psnr_model = os.path.join(weight_dir, 'best_val_psnr_model')
    save_best_val_ssim_model = os.path.join(weight_dir, 'best_val_ssim_model')

    if is_main_process:
        for folder in [weight_dir, save_bestmodel, save_lastmodel, save_best_psnr_model, 
                       save_best_ssim_model, logs_folder, save_best_val_loss_model, 
                       save_best_val_psnr_model, save_best_val_ssim_model]:
            if not os.path.exists(folder):
                os.makedirs(folder, exist_ok=True)

    # ================= 3. 数据集与 DDP 采样器 =================
    train_loader_ori, test_loader_ori = LLdataset(dir=args.data_dir, patch_size=256, batch_size=args.batch_size, 
                                                  train_file=args.train_list_file, test_file=args.test_list_file, num_workers=args.workers).get_loaders()
    train_dataset = train_loader_ori.dataset
    test_dataset = test_loader_ori.dataset

    # 使用 DistributedSampler 替代原有的 shuffle
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    # ================= 4. 模型与损失初始化 =================
    model = MGCC().to(device)
    
    if is_main_process:
        param_count = count_model_params(model)
        print(f"模型总参数量: {param_count:.2f} M") 
        
    criterion_wave = CombinedLoss().to(device)
    criterion = torch.nn.L1Loss().to(device)
    color_loss = ColorHistogramKLLoss().to(device)
    criterion_vgg = VGGPerceptualLoss().to(device)

    # 断点续训变量初始化
    lastepoch = 1
    min_loss = float('inf')
    max_psnr, max_ssim = 0, 0
    val_min_loss = float('inf')
    val_max_psnr, val_max_ssim = 0, 0

    if args.resume:
        try:
            last_info_list = process_files(save_lastmodel)[0] 
            model_name = last_info_list['文件名']
            lastepoch = int(last_info_list['epoch']) + 1
            model_path = os.path.join(save_lastmodel, model_name)    
            
            # 读取权重，去掉可能残留的 "module." 前缀
            state_dict = torch.load(model_path, map_location=device)
            new_state_dict = {}
            for k, v in state_dict.items():
                key = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[key] = v
            model.load_state_dict(new_state_dict)
            
            if is_main_process:
                print(f"从第 {lastepoch} 轮继续训练")
                
            best_info_list = process_files(save_bestmodel)[0]
            min_loss = float(best_info_list['Loss'])
            max_psnr = float(process_files(save_best_psnr_model)[0]['Loss'])
            max_ssim = float(process_files(save_best_ssim_model)[0]['Loss'])
            try: val_min_loss = float(process_files(save_best_val_loss_model)[0]['Loss'])
            except: pass
            try: val_max_psnr = float(process_files(save_best_val_psnr_model)[0]['Loss'])
            except: pass
            try: val_max_ssim = float(process_files(save_best_val_ssim_model)[0]['Loss'])
            except: pass
        except Exception as e:
            if is_main_process: print(f"加载断点失败，从头开始训练。错误: {e}")

    # 【关键】用 DDP 包装模型
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    G_opt = optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler('cuda')
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        G_opt, mode='max', factor=0.8, patience=10, threshold=1e-2, 
        threshold_mode='abs', cooldown=10, min_lr=1e-8
    )

    # ================= 5. 训练主循环 =================
    for epoch in range(lastepoch, args.num_epoch):
        train_sampler.set_epoch(epoch)  # 【关键】打乱 DDP 数据顺序
        model.train()
        
        etime = time.time()
        eloss, epsnr, essim, count = 0, 0, 0, 0
        total_L1, total_wave, total_closs, total_vgg = 0, 0, 0, 0
        
        total_batches = len(train_loader)
        
        # 仅在主进程展示 tqdm
        pbar = tqdm(enumerate(train_loader), total=total_batches, ascii=True, dynamic_ncols=True,
                    desc=f"Epoch {epoch:04d} [Loss: 0.000]", disable=not is_main_process)
                    
        for i, databatch in pbar:
            input_raw = databatch["input_raw"].to(device, non_blocking=True)
            gt_rgb = databatch["gt_rgb"].to(device, non_blocking=True)
            count += 1

            G_opt.zero_grad()
            
            with torch.amp.autocast('cuda'):
                preds, x_raw = model(input_raw)
                
                loss_wave = criterion_wave(preds, gt_rgb)
                loss_L1 = criterion(preds, gt_rgb)
                closs = color_loss(preds.float(), gt_rgb.float())
                
                loss_vgg = criterion_vgg(preds, gt_rgb)
                if isinstance(loss_vgg, torch.Tensor) and loss_vgg.dim() > 0:
                    loss_vgg = loss_vgg.mean()
                
                loss = loss_L1 + loss_wave + 2 * closs + loss_vgg

            scaler.scale(loss).backward()
            scaler.step(G_opt)
            scaler.update()
            
            # 局部指标提取
            l_val = loss.item()
            l1_val = loss_L1.item()
            wave_val = loss_wave.item()
            c_val = closs.item()
            vgg_val = loss_vgg.item()

            eloss += l_val
            total_L1 += l1_val
            total_wave += wave_val
            total_closs += c_val
            total_vgg += vgg_val
            
            preds_metric = torch.clip(preds.detach().float() * 255.0, 0, 255.0)
            gt_metric = torch.clip(gt_rgb.detach().float() * 255.0, 0, 255.0)
            epsnr += get_psnr_torch(preds_metric, gt_metric).mean().item()
            essim += get_ssim_torch(preds_metric, gt_metric).mean().item()

            if is_main_process:
                pbar.set_description(
                    f"Epoch {epoch:04d} [Loss: {l_val:.3f} | L1:{l1_val:.3f} | Wave:{wave_val:.3f} | Color:{c_val:.3f} | VGG:{vgg_val:.3f}]"
                )

        # 【关键】同步多卡指标
        metrics_tensor = torch.tensor([eloss, epsnr, essim, total_L1, total_wave, total_closs, total_vgg], device=device)
        reduced_metrics = reduce_tensor(metrics_tensor) / count

        aloss = reduced_metrics[0].item()
        apsnr = reduced_metrics[1].item()
        assim = reduced_metrics[2].item()
        avg_L1 = reduced_metrics[3].item()
        avg_wave = reduced_metrics[4].item()
        avg_closs = reduced_metrics[5].item()
        avg_vgg = reduced_metrics[6].item()

        temp_loss, temp_psnr, temp_ssim = aloss, apsnr, assim
        
        if is_main_process:
            writer.add_scalar('train_loss/total', aloss, epoch)
            writer.add_scalar('train_loss/L1', avg_L1, epoch)
            writer.add_scalar('train_loss/wavelet', avg_wave, epoch)
            writer.add_scalar('train_loss/color', avg_closs, epoch)
            writer.add_scalar('train_loss/vgg', avg_vgg, epoch)
            writer.add_scalar('train_PSNR', apsnr, epoch)
            writer.add_scalar('train_SSIM', assim, epoch)
            writer.add_scalar('learning_rate', G_opt.param_groups[0]['lr'], epoch)

            print(f"\nEpoch = {epoch}. \tTotal Loss = {aloss:.4f} | L1 = {avg_L1:.4f} | Wave = {avg_wave:.4f} | Color*2 = {2*avg_closs:.4f} | VGG = {avg_vgg:.4f}")
            print(f"\tPSNR = {apsnr:.4f}, \tSSIM = {assim:.4f},\tTime = {time.time() - etime:.2f}s")

        # ================= 6. 验证阶段 =================
        if epoch % 5 == 0:
            val_eloss, val_epsnr, val_essim = 0.0, 0.0, 0.0
            val_count = 0
            model.eval()
            with torch.no_grad():
                val_pbar = tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Epoch {epoch} 验证", 
                                ascii=True, dynamic_ncols=True, disable=not is_main_process)
                for i, databatch in val_pbar:
                    input_raw = databatch['input_raw'].to(device, non_blocking=True)
                    gt_rgb = databatch['gt_rgb'].to(device, non_blocking=True)
                    val_count += 1

                    with torch.amp.autocast('cuda'):
                        preds, x_raw = model(input_raw)
                        
                        loss_wave = criterion_wave(preds, gt_rgb)
                        loss_L1 = criterion(preds, gt_rgb)
                        closs = color_loss(preds.float(), gt_rgb.float())
                        
                        loss_vgg = criterion_vgg(preds, gt_rgb)
                        if isinstance(loss_vgg, torch.Tensor) and loss_vgg.dim() > 0:
                            loss_vgg = loss_vgg.mean()
                            
                        loss = loss_L1 + loss_wave + 2 * closs + loss_vgg
                    
                    val_eloss += loss.item()
                    
                    preds_metric = torch.clip(preds.detach().float() * 255.0, 0, 255.0)
                    gt_metric = torch.clip(gt_rgb.detach().float() * 255.0, 0, 255.0)
                    val_epsnr += get_psnr_torch(preds_metric, gt_metric).mean().item()
                    val_essim += get_ssim_torch(preds_metric, gt_metric).mean().item()

            # 同步验证指标
            val_metrics_tensor = torch.tensor([val_eloss, val_epsnr, val_essim], device=device)
            val_reduced = reduce_tensor(val_metrics_tensor) / val_count

            val_loss = val_reduced[0].item()
            val_psnr = val_reduced[1].item()
            val_ssim = val_reduced[2].item()

            scheduler.step(val_psnr)  

            if is_main_process:

                print(f"\n===== Epoch {epoch} 验证结果 =====")
                print(f"验证集 - Loss: {val_loss:.4f}, PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}")
                print("================================\n")
                
                # ------ 提取未包装的模型 state_dict ------
                model_state = model.module.state_dict()

                if val_loss < val_min_loss:
                    val_min_loss = val_loss
                    val_min_loss_epoch = epoch
                    for filename in os.listdir(save_best_val_loss_model):
                        if filename.endswith('.pth'): os.remove(os.path.join(save_best_val_loss_model, filename))
                    torch.save(model_state, os.path.join(save_best_val_loss_model, f"bestvalloss_{val_loss:.4f}_{epoch}.pth"))

                if val_psnr > val_max_psnr:
                    val_max_psnr = val_psnr
                    val_max_psnr_epoch = epoch
                    for filename in os.listdir(save_best_val_psnr_model):
                        if filename.endswith('.pth'): os.remove(os.path.join(save_best_val_psnr_model, filename))
                    torch.save(model_state, os.path.join(save_best_val_psnr_model, f"bestvalpsnr_{val_psnr:.4f}_{epoch}.pth"))

                if val_ssim > val_max_ssim:
                    val_max_ssim = val_ssim
                    val_max_ssim_epoch = epoch
                    for filename in os.listdir(save_best_val_ssim_model):
                        if filename.endswith('.pth'): os.remove(os.path.join(save_best_val_ssim_model, filename))
                    torch.save(model_state, os.path.join(save_best_val_ssim_model, f"bestvalssim_{val_ssim:.4f}_{epoch}.pth"))
            
            # 等待 rank0 判定并清理缓存
            dist.barrier()
            del preds, x_raw, loss_wave, loss_L1, closs, loss_vgg, loss
            del input_raw, gt_rgb 
            torch.cuda.empty_cache()

        # ================= 7. 训练集模型保存 (仅主进程) =================
        if is_main_process:
            model_state = model.module.state_dict()
            if temp_loss < min_loss:
                min_loss = temp_loss
                min_loss_epoch = epoch
                for f in os.listdir(save_bestmodel):
                    if f.endswith('.pth'): os.remove(os.path.join(save_bestmodel, f))
                torch.save(model_state, os.path.join(save_bestmodel, f"bestmodel_{temp_loss:.4f}_{epoch}.pth"))   
                  
            if temp_psnr > max_psnr:
                max_psnr = temp_psnr
                for f in os.listdir(save_best_psnr_model):
                    if f.endswith('.pth'): os.remove(os.path.join(save_best_psnr_model, f))
                torch.save(model_state, os.path.join(save_best_psnr_model, f"bestpsnrmodel_{temp_psnr:.4f}_{epoch}.pth")) 
            
            if temp_ssim > max_ssim:
                max_ssim = temp_ssim
                for f in os.listdir(save_best_ssim_model):
                    if f.endswith('.pth'): os.remove(os.path.join(save_best_ssim_model, f))
                torch.save(model_state, os.path.join(save_best_ssim_model, f"bestssimmodel_{temp_ssim:.4f}_{epoch}.pth")) 

            for f in os.listdir(save_lastmodel):
                if f.endswith('.pth'): os.remove(os.path.join(save_lastmodel, f))
            torch.save(model_state, os.path.join(save_lastmodel, f"ModelSnapshot_{temp_loss:.4f}_{epoch}.pth"))
        
        # 进程同步防止提前进入下一个 epoch
        dist.barrier()

    if is_main_process:
        print(f"Min Loss: {min_loss:.4f} (Epoch {min_loss_epoch}), Max PSNR: {max_psnr:.4f}, Max SSIM: {max_ssim:.4f} ")

        
    dist.destroy_process_group()

if __name__ == "__main__":
    # 【注意】移除 os.environ["CUDA_VISIBLE_DEVICES"]，让 DDP 自动分配显卡
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/root/shared-nvme/MAI')
    parser.add_argument('--train_list_file', type=str, default='/root/shared-nvme/MAI/data/ISP/AAAI-25/MAI_dataset/train.txt')
    parser.add_argument('--test_list_file',  type=str, default='/root/shared-nvme/MAI/data/ISP/AAAI-25/MAI_dataset/val.txt')
    parser.add_argument('--train_datatype', type=str, default='train')
    parser.add_argument('--test_datatype',  type=str, default='test')
    parser.add_argument('--result_dir', type=str, default='/root/shared-nvme/MSOCO/Results/result_MSO_CCA/MAINoduili/L1_WaveLoss_PerceptualLoss')
    parser.add_argument('--ps', type=int, default=512)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # 【重要提醒】：在 DDP 中，此处的 batch_size 指的是「单张显卡」上的 batch_size
    # 如果你用 4 张卡跑，这里的 16 意味着总体 batch_size = 16 * 4 = 64
    # 如果显存爆炸，请往下调 (如改为 12 或 8)
    parser.add_argument('--batch_size', type=int, default=64) 
    
    parser.add_argument('--num_epoch', type=int, default=4001)
    parser.add_argument('--model_save_freq', type=int, default=200)
    parser.add_argument('--resume', type=bool, default=True, help='continue training')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    
    train_and_evaluate(args, 0.5, 0.5)

    # torchrun --nproc_per_node=1 /root/shared-nvme/MSOCO/train_MAINoduili.py
    # /root/.cache/torch/hub/checkpoints/vgg16-397923af.pth
