import argparse
from ast import arg
import logging
import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm  # 修改导入语句
import numpy as np

# from DO_jianmo import model
from models.model_MSO_CCA_RAWLoss import *
from datasets.ZRRMCRDataset import *
from utils.metrics import get_psnr_torch, get_ssim_torch, get_lpips_torch


from PIL import Image
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
    total_params = count_parameters(model)
    print(f'Total trainable parameters: {total_params}')
    model.eval()
    print(model.w_r,model.w_g,model.w_b)
   
    # 获取迭代器的长度
    total_batches = len(test_loader)
    print(total_batches) # 598
    # 初始化 PSNR 和 SSIM 列表
    psnr_list = []
    ssim_list = []
    lpips_list = []
    # loss_fn_alex_v1 = lpips.LPIPS(net='squeeze', version='0.1').to(device)

    
    

    with torch.no_grad(): 
        # testing
        agvtime=0
        count =0
        for i, databatch in tqdm(enumerate(test_loader), total=total_batches):
            etime = time.time()
            # input_path, gt_path = data['low_RAW_img'][0], data['high_RGB_img'][0]#, data['ratio'][0]
            # index = data['index']
            input_raw = databatch['input_raw'].cuda(non_blocking=True)
            gt_rgb = databatch['gt_rgb'].cuda(non_blocking=True)
            input_path = databatch['index'][0]

            signal_time = time.time()
            pred_rgb,_ = model(input_raw)
            agvtime=agvtime+(time.time()-signal_time)
            count+=1

            pred_rgb = torch.clamp(pred_rgb, 0, 1)
            gt_rgb = torch.clamp(gt_rgb, 0, 1)
            lpips = get_lpips_torch(pred_rgb, gt_rgb)

            preds = torch.clip(pred_rgb * 255.0, 0, 255.0)
            gt_rgb = torch.clip(gt_rgb * 255.0, 0, 255.0)
            
            single_psnr = get_psnr_torch(preds, gt_rgb)
            single_ssim = get_ssim_torch(preds, gt_rgb)

            #自己算法
            # psnr = get_psnr_torch(pred_rgb, gt_rgb)
            # ssim = get_ssim_torch(pred_rgb, gt_rgb)
            # 将 PSNR 和 SSIM 添加到列表中
            psnr_list.append(single_psnr.mean().item())
            ssim_list.append(single_ssim.mean().item())
            lpips_list.append(lpips.item())
            logging.info(f"PSNR: {single_psnr.mean().item()},SSIM: {single_ssim.mean().item()},LPIPS: {lpips.item()},SIGTime: {signal_time - etime},Time: {time.time() - etime}")

            pred_rgb =  preds.round()
            # gt_rgb = gt_rgb.round()

            # print(f"outputs的大小为{outputs.shape},类型为{type(outputs)}") # outputs的大小为torch.Size([2848, 4256, 3]),类型为<class 'torch.Tensor'>
            pred_rgb = pred_rgb.squeeze(0).cpu().numpy().astype(np.uint8)
            save_path = os.path.join(images_folder, os.path.basename(input_path)) + '.png'
            Image.fromarray(pred_rgb.transpose(1, 2, 0)).save(save_path)
            
            # gt_rgb = gt_rgb.squeeze(0).cpu().numpy().astype(np.uint8)
            # save_path = os.path.join(images_folder, os.path.basename(input_path)) + '_gt.png'
            # Image.fromarray(gt_rgb.transpose(1, 2, 0)).save(save_path)
        
            
        # 计算平均 PSNR 和 SSIM
        avg_psnr = np.mean(psnr_list)
        avg_ssim = np.mean(ssim_list)
        avg_lpips = np.mean(lpips_list) 
        logging.info(f"Average PSNR: {avg_psnr},Average SSIM: {avg_ssim},Average LPIPS: {avg_lpips},Average time :{agvtime/count}")

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    
    parser = argparse.ArgumentParser(description="evaluating model")
    parser.add_argument('--gpu', type=int, default=0)

    parser.add_argument('--data_dir', type=str, default='/root/shared-nvme/ZRR')
    parser.add_argument('--train_list_file', type=str, default='/root/shared-nvme/ZRR/train.txt')
    parser.add_argument('--test_list_file',  type=str, default='/root/shared-nvme/ZRR/test.txt')
    
    parser.add_argument('--result_dir', type=str, default='/root/shared-nvme/MSOCO/test_result/trainZRR_testMAI/best_psnr_model')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1, help='multi-threads for data loading')
    parser.add_argument('--model', type=str, default='') # pretrain model path
    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    test(args)