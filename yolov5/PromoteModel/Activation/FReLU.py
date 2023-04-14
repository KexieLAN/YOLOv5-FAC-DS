import torch
import torch.nn as nn


# FReLU非线性激活函数，在只增加一点点的计算负担的情况下，将ReLU和PReLU扩展成2D激活函数。
# 具体的做法是将max()函数内的条件部分（原先ReLU的x<0部分）换成了2D的漏斗条件，
# 解决了激活函数中的空间不敏感问题，
# 使规则（普通）的卷积也具备捕获复杂的视觉布局能力，使模型具备像素级建模的能力。

# FReLU https://arxiv.org/abs/2007.11824

class FReLU(nn.Module):
    # channle_in,Kernel
    def __init__(self, c1, k=3):
        super().__init__()
        # group=1不分离通道  1*（64*64*4）
        # group=2分成两份    2*（64*64*2）
        # group=in_channel可以完全分离通道  4*(64*64*1)
        self.conv = nn.Conv2d(c1, c1, k, 1, 1, groups=c1, bias=False)
        # 全局归一化
        self.bn = nn.BatchNorm2d(c1)

    def forward(self, x):
        return torch.max(x, self.bn(self.conv(x)))
