import torch
import torchvision
import torch.nn as nn
import numpy as np



def to_one_hot_alpha(num_classes, label, alpha):
    """
    标签转独热编码
    :param num_classes: 类别数
    :param label: shape(batch_size, x, y, z) or (batch_size, h, w)
    :param alpha: shape(num_classes, ), 权重值
    :return: one_hot_label, shape(batch_size, num_classes x, y, z)
    """
    shape_ = label.shape  # [batch_size, x, y, z]
    # print(f'label:\n{label}')

    if len(shape_) == 4:
        template_size = (shape_[0], shape_[1], shape_[2], shape_[3])  # [batch_size, x, y, z]
        template_size_view = (shape_[0], 1, shape_[1], shape_[2], shape_[3])  # [batch_size, 1, x, y, z]
    elif len(shape_) == 3:
        template_size = (shape_[0], shape_[1], shape_[2])  # [batch_size, h, w]
        template_size_view = (shape_[0], 1, shape_[1], shape_[2])  # [batch_size, 1, h, w]

    one_hots = []  # 记录one_hot标签
    alpha_one_hots = []  # 记录alpha的one hot标签

    for i in range(num_classes):
        template = torch.ones(template_size)
        template_a = torch.ones(template_size) * alpha[i]

        template[label != i] = 0  # 在 label != 当前标签值的地方赋值为0

        template = template.view(template_size_view)
        template_a = template_a.view(template_size_view)

        one_hots.append(template)  # 存储当前标签
        alpha_one_hots.append(template_a)

    one_hot_label = torch.cat(one_hots, dim=1)  # 所有标签的矩阵拼接起来
    one_hot_alpha = torch.cat(alpha_one_hots, dim=1)

    return one_hot_label, one_hot_alpha




class FocalLoss(nn.Module):
    def __init__(self,
                 alpha: list = [0.2, 0.5],
                 gamma: float = 2.0,
                 num_class: int = 2,
                 reduction: str = 'mean',
                 device: str = 'cuda',
                 if_fl: bool = True):
        """
        注意，本 Focal Loss 输入的是已经softmax的outputs
        :param alpha: 权重系数列表，如三分类中第0类权重0.2，第1类权重0.3，第2类权重0.5
        :param gamma: 困难样本挖掘的gamma
        :param num_class: 用于计算的类别
        :param reduction:选择是计算均值还是和，'mean' or 'sum'
        :param device: 计算过程中的设备，输入时记得填写。
        :param if_fl:是否计算gamma部分，默认计算，即True
        """
        super(FocalLoss, self).__init__()

        assert len(alpha) == num_class
        self.alpha = torch.tensor(alpha)
        self.gamma = gamma
        self.reduction = reduction
        self.device = device
        self.if_fl = if_fl

    def forward(self, prob, label):
        """
        迭代过程……
        :param prob: 输入的模型预测概率图，如
            output = net(x);
            prob = F.softmax(dim=1)，经过softmax之后的概率
            常见的三维shape是，(batch_size, num_class, x, y, z)或者二维shape:(batch_size, num_class, h, w)
        :param label: 输入的标签，常见三维shape：(batch_size, x, y, z)或则二维shape:(batch_size, h, w)
        output: [batch_size, num_class, 220, 220, 220]
        label: [batch_size, 220, 220, 220]
        one_hot_label: [batch_size, num_class, 220, 220, 220]
        alpha_t: shape:(num_class,),  Like :[0.2, 0.3, 0.5]
        probability: F.softmax(output), shape: [batch_size, 33, 220, 220, 220]
        alpha = [1, 220, 220, 220]
        """

        # 标签和权重都进行独热编码
        one_hot_label, alpha = to_one_hot_alpha(prob.shape[1], label, self.alpha)
        one_hot_label, alpha = one_hot_label.to(self.device), alpha.to(self.device)
        # one hot label shape:(batch_size, num_class, x, y, z); alpha's shape is the same

        # print(f'label:\n{label}')
        # print(f'label.shape:{label.shape}')
        #
        # print(f'one_hot_label:\n{one_hot_label}')
        # print(f'one_hot_label.shape:{one_hot_label.shape}')
        #
        # print(f'alpha:\n{alpha}')
        # print(f'alpha.shape:{alpha.shape}')

        # cross entropy，即softmax 交叉熵
        ce_loss = torch.mul(-torch.log(prob), one_hot_label)  # ce loss shape:(batch_size, num_class, x, y, z);
        # print(f'ce_loss:\n{ce_loss}')
        # print(f'ce_loss.shape:{ce_loss.shape}')

        """
        交叉熵公式， Loss = - label · log(softmax(outputs))
        α-balanced 交叉熵， Loss = - alpha · label · log(softmax(outputs))
        Focal Loss: Loss = - (1 - p)^γ · alpha · label · log(softmax(outputs))
        """
        loss = alpha * ce_loss  # 交叉熵 ✖ α，即α-balanced 交叉熵

        # multiply (1 - pt) ^ gamma，可以将focal loss的公式理解为：FL_Loss = (1 - p)^γ * CE_Loss
        if self.if_fl:
            loss = (torch.pow((1 - prob), self.gamma)) * loss

        # print(f'loss:\n{loss}')
        # print(f'loss.shape:{loss.shape}')

        loss = loss.sum(dim=1)
        # print(f'loss:\n{loss}')
        # print(f'loss.shape:{loss.shape}')
        if self.reduction == "mean":
            return torch.mean(loss)
        if self.reduction == "sum":
            return torch.sum(loss)
        return loss
