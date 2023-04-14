import math

import torch
import torch.nn as nn

# 较为均衡的注意力机制
# 作者认为，捕获通道的依赖关系是低效且不必要的，卷积具有良好的跨通道信息获取能力
class EAC(nn.Module):
    def __init__(self, channle, b=1, gamma=2):
        super(EAC, self).__init__()
        # 求卷积核大小
        kernel_size = int(abs((math.log(channle, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2
        # 在全局平均池化后，加一个1D卷积进行学习
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 1D卷积就是横着（顺着一个维度）扫完
        self.conv1d = nn.Conv1d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, w, h = x.size()
        avg = self.avgpool(x).view([b, 1, c])
        out = self.conv1d(avg)
        out = self.sigmoid(out).view([b, c, 1, 1])
        return out * x
