import os

def process_files(directory_path):
    """
    检查指定目录中的.pth文件，并从文件名中提取loss和轮数信息。
    
    参数:
    directory_path (str): 目录路径
    
    返回:
    list: 包含每个文件的loss、文件名和epoch的列表
    """
    # 获取目录中的所有文件
    files = [f for f in os.listdir(directory_path) if f.endswith('.pth')]

    # 初始化列表，用于存储每个文件的信息
    file_info_list = []

    # 遍历文件，划分文件名并提取loss和轮数
    for file_name in files:
        # 使用split方法从文件名中提取loss和轮数
        parts = file_name.split('_')
        if len(parts) >= 3 and parts[-1].endswith('.pth'):
            loss = parts[-2]
            epoch = parts[-1].split('.')[0]

            # 存储文件的信息到列表中
            file_info_list.append({
                '文件名': file_name,
                'Loss': loss,
                'epoch': epoch
            })

    # 如果没有找到符合条件的文件，则打印提示信息
    if not file_info_list:
        print("目录中没有符合条件的.pth文件。")

    return file_info_list



if __name__ == '__main__':
    # 测试函数
    directory_path = "/raid/hbj/code/SID_dataset/train_firstsatge/unet_tensorboard_161_psnr_ssim/last_model/"
    file_info_list = process_files(directory_path)
    model_name = file_info_list['文件名']
    model_loss = file_info_list['Loss']
    model_epoch = file_info_list['epoch']
    
   
    print(f"文件名: {model_name}")
    print(f"Loss: {model_loss}")
    print(f"epoch: {model_epoch}")
    print("-" * 50)

