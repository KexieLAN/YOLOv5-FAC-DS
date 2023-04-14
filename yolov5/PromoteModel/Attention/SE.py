import torch
import torch.nn as nn
import math
import torchvision

#

# 从输入的整体通道，看每个通道的权值
# 输入 256*256*16 -> 1*1*6
class SE(nn.Module):
    # ratio：缩放比例
    def __init__(self, channel, ratio=16):
        super(SE, self).__init__()
        # 对输入的特征层进行自适应全局平均池化（高，宽）
        self.avgpooling = nn.AdaptiveAvgPool2d(1)
        self.model = nn.Sequential(
            # 两次全连接操作
            # 第一次全连接神经元个数较少，第二次全连接神经元个数和输入特征层相同。
            # 先对C个通道降维再扩展回C通道。好处就是一方面降低了网络计算量，一方面增加了网络的非线性能力。
            nn.Linear(channel, channel // ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // ratio, channel, bias=False),
            # Sigmod将输出固定在0~1之间
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # b,c,w,h  ->  b,c,1,1
        y = self.avgpooling(x).view(b, c)
        # b,c -> b,c/ratio -> b,c -> b,c,1,1
        y = self.model(y).view(b, c, 1, 1)
        return x * y

