import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms


def load_image(image_path, img_size=(256, 256)):
    """
    加载并调整图像大小
    
    参数:
        image_path: 图像文件路径
        img_size: 调整后的图像尺寸
        
    返回:
        img: 加载的PIL图像对象
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(img_size, Image.LANCZOS)
        return img
    except Exception as e:
        print(f"加载图像失败 {image_path}: {e}")
        return None


def image_to_tensor(img, normalize=True):
    """
    将PIL图像转换为PyTorch张量
    
    参数:
        img: PIL图像对象
        normalize: 是否将像素值归一化到[-1, 1]
        
    返回:
        tensor: 转换后的PyTorch张量
    """
    transform_list = [transforms.ToTensor()]
    if normalize:
        # 转换为[-1, 1]范围
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
    
    transform = transforms.Compose(transform_list)
    return transform(img)


def tensor_to_image(tensor, denormalize=True):
    """
    将PyTorch张量转换为PIL图像
    
    参数:
        tensor: PyTorch张量
        denormalize: 是否从[-1, 1]反归一化到[0, 1]
        
    返回:
        img: 转换后的PIL图像
    """
    # 确保张量在CPU上
    if tensor.device.type != 'cpu':
        tensor = tensor.cpu()
    
    # 反归一化
    if denormalize:
        tensor = tensor * 0.5 + 0.5  # 从[-1, 1]到[0, 1]
    
    # 确保值在有效范围内
    tensor = torch.clamp(tensor, 0, 1)
    
    # 转换为PIL图像
    img = transforms.ToPILImage()(tensor)
    return img


class TestDatasetLoader:
    """
    测试数据集加载器，用于加载和管理测试图像
    支持FFHQ、ImageNet和混合测试集
    """
    
    def __init__(self, data_dir="datasets", img_size=(256, 256)):
        """
        初始化测试数据集加载器
        
        参数:
            data_dir: 数据集根目录
            img_size: 图像尺寸
        """
        self.data_dir = data_dir
        self.img_size = img_size
        self.mixed_test_dir = os.path.join(data_dir, "mixed_256x256")
        self.dataset_index_path = os.path.join(data_dir, "dataset_index.txt")
        
        # 检查数据集是否存在
        self._check_dataset_exists()
        
        # 加载数据集索引
        self.image_paths = self._load_dataset_index()
        print(f"测试数据集加载完成，共包含 {len(self.image_paths)} 张图像")
    
    def _check_dataset_exists(self):
        """
        检查数据集是否存在
        """
        if not os.path.exists(self.mixed_test_dir):
            print(f"警告: 混合测试集目录不存在: {self.mixed_test_dir}")
            print("请先运行 download_datasets.py 下载测试数据")
    
    def _load_dataset_index(self):
        """
        加载数据集索引文件
        
        返回:
            image_paths: 图像文件路径列表
        """
        image_paths = []
        
        # 如果存在索引文件，优先使用索引文件
        if os.path.exists(self.dataset_index_path):
            try:
                with open(self.dataset_index_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            image_path = os.path.join(self.data_dir, line)
                            if os.path.exists(image_path):
                                image_paths.append(image_path)
                print(f"成功从索引文件加载 {len(image_paths)} 张图像")
            except Exception as e:
                print(f"读取索引文件失败: {e}")
        
        # 如果索引文件不存在或为空，则直接扫描混合测试集目录
        if not image_paths and os.path.exists(self.mixed_test_dir):
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            for filename in os.listdir(self.mixed_test_dir):
                if any(filename.lower().endswith(ext) for ext in image_extensions):
                    image_path = os.path.join(self.mixed_test_dir, filename)
                    image_paths.append(image_path)
            print(f"成功扫描混合测试集目录，找到 {len(image_paths)} 张图像")
        
        return image_paths
    
    def get_image_by_index(self, index, return_tensor=True, normalize=True):
        """
        根据索引获取图像
        
        参数:
            index: 图像索引
            return_tensor: 是否返回PyTorch张量
            normalize: 是否归一化
            
        返回:
            image: PIL图像或PyTorch张量
            image_path: 图像文件路径
        """
        if index >= len(self.image_paths):
            print(f"索引越界: {index} >= {len(self.image_paths)}")
            return None, None
        
        image_path = self.image_paths[index]
        img = load_image(image_path, self.img_size)
        
        if img is None:
            return None, image_path
        
        if return_tensor:
            img = image_to_tensor(img, normalize=normalize)
            # 添加批次维度
            img = img.unsqueeze(0)
        
        return img, image_path
    
    def get_random_images(self, num_images=5, return_tensor=True, normalize=True):
        """
        获取随机图像样本
        
        参数:
            num_images: 随机图像数量
            return_tensor: 是否返回PyTorch张量
            normalize: 是否归一化
            
        返回:
            images: 图像列表
            image_paths: 图像文件路径列表
        """
        if num_images > len(self.image_paths):
            num_images = len(self.image_paths)
            print(f"调整请求的图像数量为 {num_images}")
        
        # 随机选择索引
        indices = np.random.choice(len(self.image_paths), num_images, replace=False)
        
        images = []
        image_paths = []
        
        for idx in indices:
            img, path = self.get_image_by_index(idx, return_tensor, normalize)
            if img is not None:
                images.append(img)
                image_paths.append(path)
        
        return images, image_paths
    
    def get_all_images(self, return_tensor=True, normalize=True):
        """
        获取所有测试图像
        
        参数:
            return_tensor: 是否返回PyTorch张量
            normalize: 是否归一化
            
        返回:
            images: 图像列表
            image_paths: 图像文件路径列表
        """
        images = []
        image_paths = []
        
        for i in range(len(self.image_paths)):
            img, path = self.get_image_by_index(i, return_tensor, normalize)
            if img is not None:
                images.append(img)
                image_paths.append(path)
        
        return images, image_paths
    
    def create_batch(self, images):
        """
        创建批次张量
        
        参数:
            images: 图像张量列表
            
        返回:
            batch: 批次张量
        """
        if not images:
            return None
        
        return torch.cat(images, dim=0)


def create_test_dataset_loader(data_dir="datasets", img_size=(256, 256)):
    """
    创建测试数据集加载器的便捷函数
    
    参数:
        data_dir: 数据集目录
        img_size: 图像尺寸
        
    返回:
        dataset_loader: 测试数据集加载器实例
    """
    return TestDatasetLoader(data_dir, img_size)

# 添加缺失的函数接口以保持向后兼容性
def load_test_dataset(dataset_path, num_images=None):
    """
    加载测试数据集
    
    Args:
        dataset_path: 数据集路径
        num_images: 加载的图像数量，如果为None则加载所有图像
        
    Returns:
        图像列表和图像名称列表
    """
    # 获取所有图像文件
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    for file in os.listdir(dataset_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(os.path.join(dataset_path, file))
    
    # 排序以确保顺序一致
    image_files.sort()
    
    # 如果指定了数量，则截取
    if num_images is not None:
        image_files = image_files[:num_images]
    
    # 加载图像
    images = []
    image_names = []
    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')
        images.append(image)
        image_names.append(os.path.basename(image_file))
    
    return images, image_names

def load_test_images(dataset_path, num_images=None):
    """加载测试图像（兼容接口）"""
    return load_test_dataset(dataset_path, num_images)

def get_dataset_paths(base_dir='datasets'):
    """获取数据集路径（兼容接口）"""
    paths = {
        'ffhq': os.path.join(base_dir, 'ffhq', '256x256'),
        'imagenet': os.path.join(base_dir, 'imagenet', '256x256'),
        'mixed': os.path.join(base_dir, 'mixed_256x256')
    }
    return paths
