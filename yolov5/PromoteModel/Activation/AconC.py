import torch
import torch.nn as nn
import cv2
import mat


# 这是2021年新出的一个激活函数，
# 先从ReLU函数出发，采用Smoth maximum近似平滑公式证明了Swish就是ReLU函数的近似平滑表示，
# 这也算提出一种新颖的Swish函数解释。之后进一步分析ReLU的一般形式Maxout系列激活函数，
# 再次利用Smoth maximum将Maxout系列扩展得到简单且有效的ACON系列激活函数：ACON-A、ACON-B、ACON-C。
# 最终提出meta-ACON，动态的学习（自适应）激活函数的线性/非线性，显著提高了表现。

# https://blog.csdn.net/qq_38253797/article/details/118964626

# https://arxiv.org/abs/2009.04759

class AconC(nn.Module):
    def __init__(self, c1):
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.beta = nn.Parameter(torch.randn(1, c1, 1, 1))

    def forward(self, x):
        # AconC: (p1-p2) * x * sigmoid(bate*x*(p1-p2)) + p2 * x
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x
