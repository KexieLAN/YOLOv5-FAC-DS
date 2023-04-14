import torch
import torch.nn as nn


# 通道注意力
# 和SE相似
class channle_attention(nn.Module):
    def __init__(self, channle, ratio=16):
        super(channle_attention, self).__init__()
        # Max Pooling与Avg Pooling
        self.maxpooling = nn.AdaptiveMaxPool2d(1)
        self.avgpooling = nn.AdaptiveAvgPool2d(1)
        # 共享的全连接层
        self.model = nn.Sequential(
            nn.Linear(channle, channle // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(channle // ratio, channle, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, w, h = x.size()
        # 对通道进行最大池化和空间池化
        max_pool_out = self.maxpooling(x).view([b, c])
        avg_pool_out = self.avgpooling(x).view([b, c])
        # 分别使用全连接层进行连接
        max_fc_out = self.model(max_pool_out)
        avg_fc_out = self.model(avg_pool_out)
        # 二者合并
        out = max_fc_out + avg_fc_out
        # Sigmoid控制值在0~1之间
        out = self.sigmoid(out).view([b, c, 1, 1])
        # 权重与x各个通道相乘
        return x * out


# 空间注意力
class spacial_attention(nn.Module):
    def __init__(self, kernel_size=7):
        super(spacial_attention, self).__init__()
        # (特征图的大小-算子的size+2*padding)/步长+1
        # 保持图像的W，H不变
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # 卷积层，输入2通道，输出1通道
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 获取两个层，全局最大池化 与 全局平均池化
        max_pool_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool_out, _ = torch.mean(x, dim=1, keepdim=True)
        # 连接两个层，用于后续卷积
        pool_out = torch.cat([max_pool_out, avg_pool_out], dim=1)
        # 卷积
        out = self.conv(pool_out)
        out = self.sigmoid(out)
        return out * x


class CBAM(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=3):
        super(CBAM, self).__init__()
        self.channle_attention = channle_attention(channel, ratio)
        self.spacial_attention = spacial_attention(kernel_size)

    def forward(self, x):
        x = self.channle_attention(x)
        x = self.spacial_attention(x)
        return x
