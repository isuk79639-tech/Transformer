import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
import pickle
from datetime import datetime
import logging

# 设置随机种子
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

"""
"""

# 配置日志
def setup_logging(log_file='training.log'):
    # Create file handler with UTF-8 encoding
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# 设备配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 模型配置
class Config:
    def __init__(self):
        # 模型参数
        self.d_model = 512
        self.nhead = 8
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.dim_feedforward = 2048
        self.dropout = 0.1
        
        # 训练参数
        self.batch_size = 32  # Fixed batch size, optimized for performance
        self.learning_rate = 1e-4
        self.num_epochs = 50
        self.warmup_steps = 100  # 将根据数据集大小自动调整（约1-2个epoch或总步数的8%）
        self.max_grad_norm = 1.0
        
        # 数据参数
        self.max_length = 64
        self.vocab_size = 30000
        
        # 其他参数
        self.save_dir = 'checkpoints'
        self.log_dir = 'logs'
        self.seed = 42
        
    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    def load(self, path):
        with open(path, 'r') as f:
            config_dict = json.load(f)
        for key, value in config_dict.items():
            setattr(self, key, value)

# 创建必要的目录
def create_directories(config):
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)

if __name__ == "__main__":
    # 设置随机种子
    set_seed()
    
    # 创建配置
    config = Config()
    
    # 创建目录
    create_directories(config)
    
    # 设置日志
    logger = setup_logging()
    
    logger.info("Configuration initialized successfully")
    logger.info(f"Model parameters: d_model={config.d_model}, nhead={config.nhead}")
    logger.info(f"Training parameters: batch_size={config.batch_size}, lr={config.learning_rate}")
