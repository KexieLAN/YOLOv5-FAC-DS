import math

import torch
# import numpy as np
import torch.nn as nn


# 使用时，将其更名为bbox_iou放入到metrics.py里
# alpha=1时，认为与正常的IoU无异，alpha!=1时，进化为alpha-IoU，一般认为alpha=3时，IoU有较好的表现
# Focal——EIoU的思想可以用在其他的IoU变体上，但是Focal+SIoU存在问题，且目前尚不稳定
# gamma为Focal——EIoU的参数，默认0.5，可自行修改
# 此外，仍需要修改其他文件中的部分语句
# utils/loss.py中ComputeLoss Class中的__call__函数中修改一下：
# eps用来防治零溢事件

# GIoU：计算两框的组合与未交错区域之比
# DIoU：GIoU基础上，计算两框中间的欧氏距离
# CIoU：DIoU基础上，涉及长宽比
# EIoU：在CIOU的惩罚项基础上将纵横比的影响因子拆开分别计算目标框和锚框的长和宽
# SIoU：

def bbox_iou(box1, box2, x1y1x2y1=True, GIoU=False, DIoU=False, CIoU=False, EIoU=False,
             SIoU=False, Focal=False, alpha=1, gamma=0.5, eps=1e-7):
    box2 = box2.T
    # 获取边界框的坐标
    if x1y1x2y1:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # 将xy(中心坐标)wh(候选框的宽高)转化为xyxy(四条边的所在位置)形态
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box1[3] / 2, box2[1] + box1[3] / 2
    # 联合区域
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    # 交叠区域
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + eps
    # 求出常规的IoU
    iou = inter / union
    # 根据所选定的IoU-Loss来进行计算
    if GIoU or DIoU or EIoU or CIoU or SIoU:
        # 计算出最小闭包的宽和高
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
        if SIoU:  # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            # angle_cost = 1 - 2 * torch.pow( torch.sin(torch.arcsin(sin_alpha) - np.pi/4), 2)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            # return iou - 0.5 * (distance_cost + shape_cost)
            if Focal:  # SIoU
                return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha), torch.pow(
                    inter / (union + eps), gamma)
            else:
                return iou - torch.pow(0.5 * (distance_cost + shape_cost) + eps, alpha)
        if CIoU or DIoU or EIoU:
            # c2为能覆盖锚框和目标框的最小矩形的对角线距离。
            c2 = cw ** 2 + ch ** 2 + eps
            # 计算锚框与目标框之间的欧氏距离。
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if DIoU:
                # return iou - rho2 / c2
                if Focal:  # DIoU
                    return iou - rho2 / c2, torch.pow(inter / (union + eps), gamma)
                else:
                    return iou - rho2 / c2
            elif CIoU:
                # v用来表示长宽比的相似性
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha_ciou = v / (v - iou + (1 + eps))
                # return iou - (rho2 / c2 + v * alpha_ciou)
                if Focal:  # CIoU
                    return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha)), \
                        torch.pow(inter / (union + eps), gamma)
                else:
                    return iou - (rho2 / c2 + torch.pow(v * alpha_ciou + eps, alpha))
            elif EIoU:
                # 在CIOU的惩罚项基础上将纵横比的影响因子拆开分别计算目标框和锚框的长和宽
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                # return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)
                if Focal:  # EIoU
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2), torch.pow(inter / (union + eps), gamma)
                else:
                    return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)
        else:  # GIoU https://a.rxiv.org/pdf/1902.09630.pdf
            # c_area是他们的最小闭包（矩形）
            c_area = cw * ch + eps
            # return iou - (c_area - union) / c_area
            if Focal:  # GIoU
                return iou - torch.pow((c_area - union) / c_area + eps, alpha), torch.pow(inter / (union + eps), gamma)
            else:
                return iou - torch.pow((c_area - union) / c_area + eps, alpha)
    elif Focal:  # 未采用任何优化的，纯粹的IoU
        return iou, torch.pow(inter / (union + eps), gamma)
    else:
        return iou

# 在utils/loss.py中，找到ComputeLoss类的__call__()函数，把Regression loss中计算iou的代码，换成下面这句
# iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=False, EIoU=True)
# iou(prediction, target)
