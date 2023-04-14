import torch
import torch.nn as nn

# YOLOv5默认的激活函数

# 优点
# 1.无上界(避免过拟合)
# 2.有下界（产生更强烈的正则化效果）
# 3.平滑（处处可导，更易训练）
# 4.x<0具有非单调性（对分布具有重要意义，区别于Wish与ReLU）

# SiLU https://arxiv.org/pdf/1606.08415.pdf

class SiLu(nn.Module):
    @staticmethod
    def forward(x):
        return x * torch.sigmoid(x)
