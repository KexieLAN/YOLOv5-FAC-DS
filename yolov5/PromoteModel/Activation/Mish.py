import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plot


# 无上界，非饱和，避免了因饱和而导致梯度为0（梯度消失/梯度爆炸），进而导致训练速度大大下降；
# 有下界，在负半轴有较小的权重，可以防止ReLU函数出现的神经元坏死现象；同时可以产生更强的正则化效果；
# 自身本就具有自正则化效果，可以使梯度和函数本身更加平滑，且是每个点几乎都是平滑的，这就更容易优化而且也可以更好的泛化。随着网络越深，信息可以更深入的流动；
# x<0，保留了少量的负信息，避免了ReLU的Dying ReLU现象，这有利于更好的表达和信息流动；
# 连续可微，避免奇异点；
# 非单调。

# Mish https://github.com/digantamisra98/Mish

class Mish(nn.Module):
    def forward(x):
        return x * F.softplus(x).tanh()
