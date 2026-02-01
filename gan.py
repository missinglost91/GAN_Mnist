import torch
import torchvision.transforms as transforms
import torchvision
import os
import sys
from torch import nn
import time
from tqdm import tqdm
import yaml

# 尝试导入 IPEX，如果失败则标记为不可用
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
except ImportError:
    IPEX_AVAILABLE = False


class Distinguish_Model(nn.Module):
    """判别器模型"""
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(784,512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.model(x)

class Generate_Model(nn.Module):
    """生成器模型"""
    def __init__(self):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(128,256),
            nn.Tanh(),
            nn.Linear(256,512),
            nn.ReLU(),
            nn.Linear(512,784),
            nn.Tanh()# 输出范围[-1, 1]
        )
    def forward(self,x):
        return self.model(x)

def train(config):
    """训练函数"""
    # 从config加载参数
    device = config['device']
    num_epochs = config['train']['epochs']
    data_path = config['paths']['data_path']
    checkpoint_path = config['paths']['checkpoint_path']
    lr = config['train']['lr']  #学习率
    checkpoint_freq = config['train']['checkpoint_freq']
    generate_image_freq = config['train']['generate_image_freq']
    
    # 根据操作系统平台选择合适的配置
    platform = sys.platform
    if platform.startswith('win'):
        platform_key = 'windows'
    elif platform.startswith('linux'):
        platform_key = 'linux'
    else:
        platform_key = 'macos'
        
    batch_size = config['train']['batch_size'][platform_key]
    num_workers = config['dataloader']['num_workers'][platform_key]

    print("********************************************")
    print(f"-------------将使用设备: {device}----------------")
    print(f"批处理大小 (Batch Size): {batch_size}, 数据加载线程数 (Num Workers): {num_workers}")
    print("********************************************")

    # 数据预处理
    transformer=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # 加载数据集
    train_data = torchvision.datasets.MNIST(data_path, train=True, transform=transformer, download=True)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if device != 'cpu' else False)

    # 初始化模型和优化器
    D=Distinguish_Model().to(device)
    G=Generate_Model().to(device)
    D_optimizer=torch.optim.Adam(D.parameters(),lr=lr)
    G_optimizer=torch.optim.Adam(G.parameters(),lr=lr)

    # 检查并加载检查点 (实现断点续训)
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        print(f"--- 发现检查点文件，正在加载: {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path)
        G.load_state_dict(checkpoint['g_model_state_dict'])
        D.load_state_dict(checkpoint['d_model_state_dict'])
        G_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        D_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"--- 检查点加载完毕，将从第 {start_epoch + 1} 轮开始训练 ---")
    else:
        print("--- 未发现检查点文件，将从头开始训练 ---")

    # 如果使用Intel XPU，使用IPEX进行优化
    if device == 'xpu' and IPEX_AVAILABLE:
        print("IPEX is available. Optimizing models and optimizers for XPU.")
        D, D_optimizer = ipex.optimize(D, optimizer=D_optimizer)
        G, G_optimizer = ipex.optimize(G, optimizer=G_optimizer)

    loss_fn = nn.BCELoss().to(device)

    print("\n--- 开始训练 ---")
    overall_start_time = time.time()

    # 训练循环
    for current_epoch in range(start_epoch, num_epochs):
        dis_loss_all=0
        gen_loss_all=0
        load_len=len(train_dataloader)

        print("------第 {}/{} 轮训练开始------".format(current_epoch + 1, num_epochs))

        D.train()
        G.train()

        for step,data in tqdm(enumerate(train_dataloader), desc=f"第{current_epoch + 1}轮训练", total=load_len):
            sample,label=data
            sample=sample.reshape(-1,784).to(device)
            sample_shape=sample.shape[0]

            # 训练判别器
            noise=torch.normal(mean=0,std=1,size=(sample_shape,128)).to(device)
            D_optimizer.zero_grad()
            
            Dis_true=D(sample)
            true_loss=loss_fn(Dis_true,torch.ones_like(Dis_true))
            true_loss.backward()

            fake_sample=G(noise)
            Dis_fake=D(fake_sample.detach())#它的作用是：“切断”反向传播的梯度流。
            fake_loss=loss_fn(Dis_fake,torch.zeros_like(Dis_fake))
            fake_loss.backward()
            
            Dis_loss = true_loss + fake_loss
            D_optimizer.step()

            # 训练生成器
            G_optimizer.zero_grad()
            Dis_G=D(fake_sample)
            G_loss=loss_fn(Dis_G,torch.ones_like(Dis_G))
            G_loss.backward()
            G_optimizer.step()

            with torch.no_grad():
                dis_loss_all+=Dis_loss.item()
                gen_loss_all+=G_loss.item()

        # 打印每轮的平均损失
        with torch.no_grad():
            avg_dis_loss = dis_loss_all / load_len
            avg_gen_loss = gen_loss_all / load_len
            print("第{}轮训练，判别器损失：{}，生成器损失：{}".format(current_epoch + 1, avg_dis_loss, avg_gen_loss))
            
            #with torch.no_grad(): 是一个非常有用的优化工具。当您执行任何不需要反向传播的操作时（例如：模型推理、性能评估、数据可视化、或像这里一样打印日志），都应该把它包裹在这个上下文管理器中。这是一种标准的、良好的编程习惯。
            
        # 按频率保存检查点
        if (current_epoch + 1) % checkpoint_freq == 0:
            print(f"--- 正在保存第 {current_epoch + 1} 轮的检查点 ---")
            torch.save({
                'epoch': current_epoch,
                'g_model_state_dict': G.state_dict(),
                'd_model_state_dict': D.state_dict(),
                'g_optimizer_state_dict': G_optimizer.state_dict(),
                'd_optimizer_state_dict': D_optimizer.state_dict(),
            }, checkpoint_path)
            print(f"第{current_epoch + 1}轮检查点已保存到: {checkpoint_path}")

        # 按频率生成并保存图片
        if (current_epoch + 1) % generate_image_freq == 0:
            print(f"--- 在第 {current_epoch + 1} 轮结束时生成样本图片 ---")
            generate(config, epoch=current_epoch + 1)

    total_time = time.time() - overall_start_time
    print(f"\n训练完成! 总耗时: {total_time // 3600:.0f}小时 {total_time % 3600 // 60:.0f}分钟 {total_time % 60:.0f}秒")

def generate(config, epoch=None):
    """生成图片函数"""
    # 从config加载参数
    device = config['device']
    checkpoint_path = config['paths']['checkpoint_path']
    img_save_path = config['paths']['results_path']
    num_images = config['generate']['num_images']

    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)

    try:
        model_G = Generate_Model().to(device)
        print(f"--- 正在从检查点加载生成器模型: {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path)
        model_G.load_state_dict(checkpoint['g_model_state_dict'])
        model_G.eval()
        print("--- 生成器模型加载完毕 ---")
    except FileNotFoundError:
        print(f"错误：找不到检查点文件 '{checkpoint_path}'。请先进行训练。")
        return
    except KeyError:
        print(f"错误：检查点文件 '{checkpoint_path}' 格式不正确或不包含模型。")
        return

    with torch.no_grad():
        # 创建噪声并生成图片
        noise = torch.randn(num_images, 128).to(device)
        fake_images = model_G(noise)
        # 将图片形状从 (batch, 784) 调整为 (batch, 1, 28, 28) 以便保存
        fake_images = fake_images.reshape(num_images, 1, 28, 28)

    # 根据是否传入epoch来构造文件名
    if epoch is not None:
        save_name = f"result_epoch_{epoch}.png"
    else:
        save_name = "result.png"
    
    save_path = os.path.join(img_save_path, save_name)

    # 保存图片
    torchvision.utils.save_image(
        fake_images,
        save_path,
        nrow=int(num_images**0.5),
        normalize=True
    )
    print(f"生成图片已保存到: {save_path}")


if __name__=="__main__":
    
    # 获取当前脚本目录
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        current_dir = os.getcwd()

    # 加载配置文件
    with open(os.path.join(current_dir, "config.yaml"), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 根据优先级和可用性选择设备
    device_priorities = config['device_priority']
    device = None
    for dev_type in device_priorities:
        if dev_type == 'xpu' and IPEX_AVAILABLE and hasattr(torch, 'xpu') and torch.xpu.is_available():
            device = torch.device("xpu")
            break
        elif dev_type == 'cuda' and torch.cuda.is_available():
            device = torch.device("cuda")
            torch.backends.cudnn.benchmark = True
            break
    if device is None:
        device = torch.device("cpu")
    
    config['device'] = str(device)

    # 根据设备类型动态设置检查点和结果的保存目录
    if device.type == 'cuda':
        checkpoint_dir = os.path.join(current_dir, "checkpoints_cuda")
    elif device.type == 'xpu':
        checkpoint_dir = os.path.join(current_dir, "checkpoints_xpu")
    else:
        checkpoint_dir = os.path.join(current_dir, "checkpoints_cpu")

    results_path = os.path.join(current_dir, config['paths']['results_dir'])
    os.makedirs(checkpoint_dir, exist_ok=True) #确保用于存放“检查点文件”和“结果图片”的文件夹都存在。如果不存在，就创建它们
    os.makedirs(results_path, exist_ok=True)

    # 动态构建并填充路径到config字典中，供后续函数使用
    config['paths']['data_path'] = os.path.join(current_dir, config['paths']['dataset_dir'])
    config['paths']['checkpoint_dir'] = checkpoint_dir
    config['paths']['checkpoint_path'] = os.path.join(checkpoint_dir, "checkpoint.pth")
    config['paths']['results_path'] = results_path
    
    print(f"配置文件加载完毕。将在设备 '{config['device']}' 上以 '{config['mode']}' 模式运行。")
    
    # 根据模式选择执行训练或生成
    if config['mode'] == 'train':
        train(config)
    elif config['mode'] == 'generate':
        generate(config)