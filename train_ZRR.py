import argparse
from ast import arg
import os
from telnetlib import PRAGMA_HEARTBEAT
import time
import torch
from torch import optim
import torch.nn.functional as F
import os, time, scipy.io, scipy.misc
import scipy
from torchmetrics.functional import peak_signal_noise_ratio, structural_similarity_index_measure
from torch.utils.data import DataLoader
from utils.metrics import get_psnr_torch, get_ssim_torch, get_lpips_torch
from utils.util import *
from models.model_MSO_CCA_RAWLoss import *
from Losses.wavelet_loss import CombinedLoss
from Losses.color_loss import ColorHistogramKLLoss
from datasets.ZRRMCRDataset import *
from PIL import Image
from tqdm import tqdm
#色彩一致性损失
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
    
    # save_val_model = os.path.join(weight_dir, 'val_model')
    # if not os.path.exists(save_val_model):
    #     os.makedirs(save_val_model)
    
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
    
    
    train_loader,test_loader = LLdataset(dir=args.data_dir,patch_size=256,batch_size=args.batch_size,train_file=args.train_list_file, 
                                         test_file=args.test_list_file,num_workers=4).get_loaders()

    # trainset = SIDSonyDataset(data_dir=args.data_dir,image_list_file='Sony_train_list.txt',
    #                          split='train',patch_size=args.ps)
    # train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    # testset = SIDSonyDataset(data_dir='/raid/hbj/datas/SID/Sony',image_list_file='Sony_test_list.txt',
    #                          split='test',patch_size=512)
    # test_loader = DataLoader(testset, batch_size=2, shuffle=False, num_workers=8)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MGCC().to(device)
    criterion_wave = CombinedLoss().cuda()
    criterion = torch.nn.L1Loss().cuda()
    color_loss = ColorHistogramKLLoss().cuda()
    
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
   
    if args.resume:   # 如果为True，则表示接着训练；如果为False，则表示从头开始训练
        last_info_list = process_files(save_lastmodel)[0] # 获取列表中的第一个字典
        model_name = last_info_list['文件名']
        model_loss = float(last_info_list['Loss'])
        model_epoch = int(last_info_list['epoch'])  # 将字符串转换为整数
        model_path = os.path.join(save_lastmodel, model_name)   

        state_dict = torch.load(model_path)
    # 2. 去除所有键的"module."前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            # 截取键中"module."之后的部分（如果存在）
            key = k.replace("module.", "") if k.startswith("module.") else k
            new_state_dict[key] = v
        # 3. 加载处理后的权重
        model.load_state_dict(new_state_dict) 

        # model.load_state_dict(torch.load(model_path))
        model = model.to(device)  # 确保模型在GPU上
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
        # val_min_loss = float('inf')  # 初始化最小损失为正无穷
        max_psnr = 0
        max_ssim = 0

             # 新增验证集最佳指标初始化
        val_min_loss = float('inf')  # 验证集最小损失（越小越好）
        val_max_psnr = 0  # 验证集最大PSNR（越大越好）
        val_max_ssim = 0  # 验证集最大SSIM（越大越好

    print(f"lastepoch为 {lastepoch}")



    G_opt = optim.Adam(model.parameters(), lr=args.lr)



    # 初始化学习率调度器：当loss长时间不降低时，学习率×0.8
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        G_opt,
        mode='max',          # 监控指标（psnr）越大越好
        factor=0.8,          # 衰减因子：学习率 = 学习率 × 0.8
        patience=10,         # 容忍20轮psnr不增长
        threshold=1e-2,      # loss增长超过该阈值才认为是“改善”
        threshold_mode='abs',# 绝对阈值（而非相对阈值）
        cooldown=10,          # 衰减后冷却10轮，避免频繁衰减
        min_lr=1e-8,         # 学习率下限（防止过低）
        verbose=True         # 打印学习率调整信息
    )

    for epoch in range(lastepoch,args.num_epoch):
        model.train()
        # 如果当前时期的结果目录已经存在，跳过当前时期的训练，继续下一个时期的训练。
        if os.path.isdir(weight_dir + '%04d' % epoch):
            continue
        #Calculating total loss
        etime = time.time()
        eloss = 0
        epsnr = 0
        essim = 0
        count = 0
        
        # 随机排列训练数据的索引，以确保随机性 开始遍历训练数据，对每一张图像进行训练
        # for i, databatch in enumerate(train_loader):
        total_batches = len(train_loader)
        for i, databatch in tqdm(enumerate(train_loader), total=total_batches):

            input_raw = databatch["input_raw"].cuda(non_blocking=True)

            gt_rgb = databatch["gt_rgb"].cuda(non_blocking=True)

            count += 1

            
            # 将第一阶段的输出传递给第二阶段模型
            preds,_ = model(input_raw)
            
            loss_wave = criterion_wave(preds, gt_rgb)

            loss_L1 = criterion(preds, gt_rgb)

            # loss = loss_L1
            closs = color_loss(preds,gt_rgb)
            loss = loss_L1 + loss_wave + 2*closs

            
            G_opt.zero_grad()
            loss.backward()
            G_opt.step()
            
            eloss = eloss + loss.item()   # Total Loss


            preds = torch.clip(preds * 255.0, 0, 255.0)
            gt_rgb = torch.clip(gt_rgb * 255.0, 0, 255.0)
            
            single_psnr = get_psnr_torch(preds, gt_rgb)
            single_ssim = get_ssim_torch(preds, gt_rgb)
            epsnr = epsnr + single_psnr.mean().item()  # Total psnr
            essim = essim + single_ssim.mean().item()  # Total SSIM
            
            
        # 更新学习率
        aloss = eloss/count
        apsnr = epsnr/count
        assim = essim/count

        temp_loss = aloss
        temp_psnr = apsnr
        temp_ssim = assim

        print(f"\nEpoch = {epoch}. \tLoss = {aloss}, \tPSNR = {apsnr}, \tSSIM = {assim},\tTime = {time.time() - etime}")

        torch.cuda.empty_cache()  # 手动清理GPU缓存

        # 训练循环结束后，进入验证阶段
        if epoch % 5 == 0:  # 每10个epoch验证一次
            # 重置验证阶段变量
            val_eloss = 0.0
            val_epsnr = 0.0
            val_essim = 0.0
            val_count = 0
            model.eval()  # 切换为评估模式
            with torch.no_grad():
                total_batches = len(test_loader)
                for i, databatch in tqdm(enumerate(test_loader), total=total_batches, desc=f"Epoch {epoch} 验证"):
                    input_raw = databatch['input_raw'].cuda(non_blocking=True)
                    # gt_raw = databatch['gt_raw'].cuda(non_blocking=True)
                    gt_rgb = databatch['gt_rgb'].cuda(non_blocking=True)
                    
                    val_count += 1
                    # 模型推理
                    preds, x_raw = model(input_raw)
  
                    
                    # 计算验证损失
                    loss_wave = criterion_wave(preds, gt_rgb)
                    loss_L1 = criterion(preds, gt_rgb)
                    closs = color_loss(preds, gt_rgb)
                    loss = loss_L1 + loss_wave + 2 * closs
                    
                    # 累加验证指标
                    val_eloss += loss.item()
                    preds = torch.clip(preds * 255.0, 0, 255.0)
                    gt_rgb = torch.clip(gt_rgb * 255.0, 0, 255.0)
                    # warpRGB = torch.clip(warpRGB * 255.0, 0, 255.0)
                    val_epsnr += get_psnr_torch(preds, gt_rgb).mean().item()
                    val_essim += get_ssim_torch(preds, gt_rgb).mean().item()


            # 计算验证集平均指标
            val_loss = val_eloss / val_count
            val_psnr = val_epsnr / val_count
            val_ssim = val_essim / val_count


            scheduler.step(val_psnr)  

            # 打印验证信息
            print(f"\n===== Epoch {epoch} 验证结果 =====")
            print(f"验证样本数: {val_count}")
            print(f"验证集 - Loss: {val_loss:.4f}, PSNR: {val_psnr:.4f}, SSIM: {val_ssim:.4f}")
            print("================================\n")

            # 1. 验证集最小损失模型
            if val_loss < val_min_loss:
                val_min_loss = val_loss
                val_min_loss_epoch = epoch
                # 删除旧模型，保存新模型
                for filename in os.listdir(save_best_val_loss_model):
                    if filename.endswith('.pth'):
                        os.remove(os.path.join(save_best_val_loss_model, filename))
                best_val_loss_model = os.path.join(save_best_val_loss_model, f"bestvalloss_{val_loss:.4f}_{epoch}.pth")
                torch.save(model.state_dict(), best_val_loss_model)
                print(f"更新验证集最佳Loss模型（Epoch {epoch}，Loss: {val_loss:.4f}）")

            # 2. 验证集最大PSNR模型
            if val_psnr > val_max_psnr:
                val_max_psnr = val_psnr
                val_max_psnr_epoch = epoch
                for filename in os.listdir(save_best_val_psnr_model):
                    if filename.endswith('.pth'):
                        os.remove(os.path.join(save_best_val_psnr_model, filename))
                best_val_psnr_model = os.path.join(save_best_val_psnr_model, f"bestvalpsnr_{val_psnr:.4f}_{epoch}.pth")
                torch.save(model.state_dict(), best_val_psnr_model)
                print(f"更新验证集最佳PSNR模型（Epoch {epoch}，PSNR: {val_psnr:.4f}）")

            # 3. 验证集最大SSIM模型
            if val_ssim > val_max_ssim:
                val_max_ssim = val_ssim
                val_max_ssim_epoch = epoch
                for filename in os.listdir(save_best_val_ssim_model):
                    if filename.endswith('.pth'):
                        os.remove(os.path.join(save_best_val_ssim_model, filename))
                best_val_ssim_model = os.path.join(save_best_val_ssim_model, f"bestvalssim_{val_ssim:.4f}_{epoch}.pth")
                torch.save(model.state_dict(), best_val_ssim_model)
                print(f"更新验证集最佳SSIM模型（Epoch {epoch}，SSIM: {val_ssim:.4f}）")
            # 验证结束后，强制释放变量并清理显存
            del val_eloss, val_epsnr, val_essim, val_count
            del preds, x_raw, loss_wave, loss_L1, closs, loss  # 删除验证阶段的中间张量
            del input_raw, gt_rgb  # 删除输入输出张量
            torch.cuda.empty_cache()  # 手动清理GPU缓存


        
        if temp_loss < min_loss:
            min_loss = temp_loss
            min_loss_epoch = epoch
            # 保存最好的权重
            if os.path.exists(save_bestmodel):
                # 先删除再进行保存
                for filename in os.listdir(save_bestmodel):
                    if filename.endswith('.pth'):
                        # 删除该文件
                        file_path = os.path.join(save_bestmodel, filename)
                        os.remove(file_path)
                bestmodel = os.path.join(save_bestmodel, "bestmodel_{}_{}.pth".format(temp_loss, epoch))
                torch.save(model.state_dict(), bestmodel)   
                  
        if temp_psnr > max_psnr:
            max_psnr = temp_psnr
            # 保存最好的权重
            if os.path.exists(save_best_psnr_model):
                # 先删除再进行保存
                for filename in os.listdir(save_best_psnr_model):
                    if filename.endswith('.pth'):
                        # 删除该文件
                        file_path = os.path.join(save_best_psnr_model, filename)
                        os.remove(file_path)
                bestpsnrmodel = os.path.join(save_best_psnr_model, "bestpsnrmodel_{}_{}.pth".format(temp_psnr, epoch))
                torch.save(model.state_dict(), bestpsnrmodel) 
        
        if temp_ssim > max_ssim:
            max_ssim = temp_ssim
            # 保存最好的权重
            if os.path.exists(save_best_ssim_model):
                # 先删除再进行保存
                for filename in os.listdir(save_best_ssim_model):
                    if filename.endswith('.pth'):
                        # 删除该文件
                        file_path = os.path.join(save_best_ssim_model, filename)
                        os.remove(file_path)
                bestssimmodel = os.path.join(save_best_ssim_model, "bestssimmodel_{}_{}.pth".format(temp_ssim, epoch))
                torch.save(model.state_dict(), bestssimmodel) 
        

        # 保存最后一轮的权重
        # 检查目录中是否有.pth结尾的文件
        if os.path.exists(save_lastmodel):
            # 先删除再进行保存
            for filename in os.listdir(save_lastmodel):
                if filename.endswith('.pth'):
                    # 删除该文件
                    file_path = os.path.join(save_lastmodel, filename)
                    os.remove(file_path)
            lastmodel = os.path.join(save_lastmodel, "ModelSnapshot_{}_{}.pth".format(temp_loss, epoch))
            torch.save(model.state_dict(), lastmodel)
    # 在训练结束后输出最小损失和最大PSNR
    print(f"Min Loss: {min_loss} (Epoch {min_loss_epoch}), Max PSNR: {max_psnr},Max SSIM: {max_ssim} ")
 

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/root/shared-nvme/zpc/SOCODE/ZRR')
    parser.add_argument('--train_list_file', type=str, default='/root/shared-nvme/zpc/SOCODE/ZRR/train.txt')
    parser.add_argument('--test_list_file',  type=str, default='/root/shared-nvme/zpc/SOCODE/ZRR/test.txt')
    
    parser.add_argument('--train_datatype', type=str, default='train')
    parser.add_argument('--test_datatype',  type=str, default='test')
    parser.add_argument('--result_dir', type=str, default='/root/shared-nvme/zpc/SOCODE/Results/result_MSO_ECCA/ZRR/L1_WaveLoss_sum_weights')
    parser.add_argument('--ps', type=int, default=512)
    parser.add_argument('--save_freq', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=16)#三卡72，双卡48
    parser.add_argument('--num_epoch', type=int, default=4001)
    parser.add_argument('--model_save_freq', type=int, default=200)
    parser.add_argument('--resume', type=bool, default=True, help='continue training')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    
    train_and_evaluate(args, 0.5, 0.5)
