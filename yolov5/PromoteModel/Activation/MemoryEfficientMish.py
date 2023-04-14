import torch
import torch.nn as nn
import torch.nn.functional as F


# 一种高效的Mish激活函数 不采用自动求导(自己写前向传播和反向传播) 更高效，Mish的升级版。

class MemoryEfficientMish(nn.Module):
    class F(torch.autograd.functional):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            return x.mul(torch.tanh(F.softplus(x)))

        @staticmethod
        def backward(ctx, grad_output):
            x = ctx.saved_tensors[0]
            sx = torch.sigmoid(x)
            fx = F.softplus(x).tanh()
            return grad_output * (fx + x * sx * (1 - fx * fx))

    def forward(self, x):
        return self.F.apply(x)
