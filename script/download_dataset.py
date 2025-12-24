"""
Speech Commands 数据集下载脚本
运行此脚本以下载 Google Speech Commands 数据集
"""
import os
from torchaudio.datasets import SPEECHCOMMANDS

ROOT = "./speech_commands"

def download_speech_commands():
    """下载 Speech Commands 数据集"""
    print("正在检查/下载 Speech Commands 数据集...")
    print(f"数据集保存路径: {os.path.abspath(ROOT)}")
    
    # 下载数据集
    dataset = SPEECHCOMMANDS(root=ROOT, download=True)
    
    # 获取数据集信息
    labels = sorted({datapoint[2] for datapoint in dataset})
    
    print(f"\n下载完成！")
    print(f"数据集大小: {len(dataset)} 个样本")
    print(f"类别数量: {len(labels)}")
    print(f"类别列表: {labels}")

if __name__ == "__main__":
    download_speech_commands()

