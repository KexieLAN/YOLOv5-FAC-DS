import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plot


# HandsWish(x):
# if x<=-3
#   return 0
# else if x>=+3
#   retuen x
# else
#   return x*(x+3)/6

class HardsWish(nn.Module):
    def forward(x):
        # Pytorch已经支持HardsWish，直接用
        return F.hardswish(x)

