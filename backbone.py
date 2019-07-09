# This code is modified from
# https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from torch.autograd import Variable
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
# Basic ResNet model


def cosine(x, y, scale: int = 10):
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    if len(x.size()) == 2:
        x = x.unsqueeze(1).expand(n, m, d)
        y = y.unsqueeze(0).expand(n, m, d)
        return F.softmax(scale * F.cosine_similarity(x, y, dim=2), 1)
    elif len(x.size()) == 3:
        f = x.size(2)
        x = x.unsqueeze(1).expand(n, m, d, f)
        y.unsqueeze_(0)
        y = y.unsqueeze(3).expand(n, m, d, f)
        return F.softmax(scale * F.cosine_similarity(x, y, dim=2), 1).mean(2)


def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class distLinear(nn.Module):
    def __init__(self, indim, outdim):
        super(distLinear, self).__init__()
        self.L = nn.Linear(indim, outdim, bias=False)
        self.class_wise_learnable_norm = True  # See the issue#4&8 in the github
        if self.class_wise_learnable_norm:
            # split the weight update component to direction and norm
            WeightNorm.apply(self.L, 'weight', dim=0)

        if outdim <= 200:
            # a fixed scale factor to scale the output of cos value into a
            # reasonably large input for softmax
            self.scale_factor = 2
        else:
            # in omniglot, a larger scale factor is required to handle >1000
            # output classes.
            self.scale_factor = 10

    def forward(self, x):
        x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
        x_normalized = x.div(x_norm + 0.00001)
        if not self.class_wise_learnable_norm:
            L_norm = torch.norm(
                self.L.weight.data,
                p=2,
                dim=1).unsqueeze(1).expand_as(
                self.L.weight.data)
            self.L.weight.data = self.L.weight.data.div(L_norm + 0.00001)
        # matrix product by forward function, but when using WeightNorm, this
        # also multiply the cosine distance by a class-wise learnable norm, see
        # the issue#4&8 in the github
        cos_dist = self.L(x_normalized)
        scores = self.scale_factor * (cos_dist)

        return scores


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Linear_fw(nn.Linear):  # used in MAML to forward input with fast weight
    def __init__(self, in_features, out_features):
        super(Linear_fw, self).__init__(in_features, out_features)
        self.weight.fast = None  # Lazy hack to add fast weight link
        self.bias.fast = None

    def forward(self, x):
        if self.weight.fast is not None and self.bias.fast is not None:
            # weight.fast (fast weight) is the temporaily adapted weight
            out = F.linear(x, self.weight.fast, self.bias.fast)
        else:
            out = super(Linear_fw, self).forward(x)
        return out


class Conv2d_fw(nn.Conv2d):  # used in MAML to forward input with fast weight
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            bias=True):
        super(
            Conv2d_fw,
            self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias)
        self.weight.fast = None
        if self.bias is not None:
            self.bias.fast = None

    def forward(self, x):
        if self.bias is None:
            if self.weight.fast is not None:
                out = F.conv2d(
                    x,
                    self.weight.fast,
                    None,
                    stride=self.stride,
                    padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)
        else:
            if self.weight.fast is not None and self.bias.fast is not None:
                out = F.conv2d(
                    x,
                    self.weight.fast,
                    self.bias.fast,
                    stride=self.stride,
                    padding=self.padding)
            else:
                out = super(Conv2d_fw, self).forward(x)

        return out


class BatchNorm2d_fw(
        nn.BatchNorm2d):  # used in MAML to forward input with fast weight
    def __init__(self, num_features):
        super(BatchNorm2d_fw, self).__init__(num_features)
        self.weight.fast = None
        self.bias.fast = None

    def forward(self, x):
        running_mean = torch.zeros(x.data.size()[1]).cuda()
        running_var = torch.ones(x.data.size()[1]).cuda()
        if self.weight.fast is not None and self.bias.fast is not None:
            out = F.batch_norm(
                x,
                running_mean,
                running_var,
                self.weight.fast,
                self.bias.fast,
                training=True,
                momentum=1)
            # batch_norm momentum hack: follow hack of Kate Rakelly in
            # pytorch-maml/src/layers.py
        else:
            out = F.batch_norm(
                x,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                training=True,
                momentum=1)
        return out

# Simple Conv Block


class ConvBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C = Conv2d_fw(indim, outdim, 3, padding=padding)
            self.BN = BatchNorm2d_fw(outdim)
        else:
            self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
            self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out

# Simple ResNet Block


class SimpleBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, half_res):
        super(SimpleBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(
                indim,
                outdim,
                kernel_size=3,
                stride=2 if half_res else 1,
                padding=1,
                bias=False)
            self.BN1 = BatchNorm2d_fw(outdim)
            self.C2 = Conv2d_fw(
                outdim,
                outdim,
                kernel_size=3,
                padding=1,
                bias=False)
            self.BN2 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(
                indim,
                outdim,
                kernel_size=3,
                stride=2 if half_res else 1,
                padding=1,
                bias=False)
            self.BN1 = nn.BatchNorm2d(outdim)
            self.C2 = nn.Conv2d(
                outdim,
                outdim,
                kernel_size=3,
                padding=1,
                bias=False)
            self.BN2 = nn.BatchNorm2d(outdim)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C1, self.C2, self.BN1, self.BN2]

        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need
        # a 1x1 convolution
        if indim != outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(
                    indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = BatchNorm2d_fw(outdim)
            else:
                self.shortcut = nn.Conv2d(
                    indim, outdim, 1, 2 if half_res else 1, bias=False)
                self.BNshortcut = nn.BatchNorm2d(outdim)

            self.parametrized_layers.append(self.shortcut)
            self.parametrized_layers.append(self.BNshortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu1(out)
        out = self.C2(out)
        out = self.BN2(out)
        short_out = x if self.shortcut_type == 'identity' else self.BNshortcut(
            self.shortcut(x))
        out = out + short_out
        out = self.relu2(out)
        return out


# Bottleneck block
class BottleneckBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, half_res):
        super(BottleneckBlock, self).__init__()
        bottleneckdim = int(outdim / 4)
        self.indim = indim
        self.outdim = outdim
        if self.maml:
            self.C1 = Conv2d_fw(
                indim,
                bottleneckdim,
                kernel_size=1,
                bias=False)
            self.BN1 = BatchNorm2d_fw(bottleneckdim)
            self.C2 = Conv2d_fw(
                bottleneckdim,
                bottleneckdim,
                kernel_size=3,
                stride=2 if half_res else 1,
                padding=1)
            self.BN2 = BatchNorm2d_fw(bottleneckdim)
            self.C3 = Conv2d_fw(
                bottleneckdim,
                outdim,
                kernel_size=1,
                bias=False)
            self.BN3 = BatchNorm2d_fw(outdim)
        else:
            self.C1 = nn.Conv2d(
                indim,
                bottleneckdim,
                kernel_size=1,
                bias=False)
            self.BN1 = nn.BatchNorm2d(bottleneckdim)
            self.C2 = nn.Conv2d(
                bottleneckdim,
                bottleneckdim,
                kernel_size=3,
                stride=2 if half_res else 1,
                padding=1)
            self.BN2 = nn.BatchNorm2d(bottleneckdim)
            self.C3 = nn.Conv2d(
                bottleneckdim,
                outdim,
                kernel_size=1,
                bias=False)
            self.BN3 = nn.BatchNorm2d(outdim)

        self.relu = nn.ReLU()
        self.parametrized_layers = [
            self.C1,
            self.BN1,
            self.C2,
            self.BN2,
            self.C3,
            self.BN3]
        self.half_res = half_res

        # if the input number of channels is not equal to the output, then need
        # a 1x1 convolution
        if indim != outdim:
            if self.maml:
                self.shortcut = Conv2d_fw(
                    indim, outdim, 1, stride=2 if half_res else 1, bias=False)
            else:
                self.shortcut = nn.Conv2d(
                    indim, outdim, 1, stride=2 if half_res else 1, bias=False)

            self.parametrized_layers.append(self.shortcut)
            self.shortcut_type = '1x1'
        else:
            self.shortcut_type = 'identity'

        for layer in self.parametrized_layers:
            init_layer(layer)

    def forward(self, x):

        short_out = x if self.shortcut_type == 'identity' else self.shortcut(x)
        out = self.C1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.C2(out)
        out = self.BN2(out)
        out = self.relu(out)
        out = self.C3(out)
        out = self.BN3(out)
        out = out + short_out

        out = self.relu(out)
        return out


class ConvNet(nn.Module):
    def __init__(self, depth, flatten=True):
        super(ConvNet, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            # only pooling for fist 4 layers
            B = ConvBlock(indim, outdim, pool=(i < 4))
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 1600

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNetNopool(
        nn.Module):  # Relation net use a 4 layer conv with pooling in only first two layers, else no pooling
    def __init__(self, depth):
        super(ConvNetNopool, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            # only first two layer has pooling and no padding
            B = ConvBlock(indim, outdim, pool=(
                i in [0, 1]), padding=0 if i in[0, 1] else 1)
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64, 19, 19]

    def forward(self, x):
        out = self.trunk(x)
        return out


class ConvNetS(
        nn.Module):  # For omniglot, only 1 input channel, output dim is 64
    def __init__(self, depth, flatten=True):
        super(ConvNetS, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            # only pooling for fist 4 layers
            B = ConvBlock(indim, outdim, pool=(i < 4))
            trunk.append(B)

        if flatten:
            trunk.append(Flatten())

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = 64

    def forward(self, x):
        out = x[:, 0:1, :, :]  # only use the first dimension
        out = self.trunk(out)
        return out


class ConvNetSNopool(
        nn.Module):  # Relation net use a 4 layer conv with pooling in only first two layers, else no pooling. For omniglot, only 1 input channel, output dim is [64,5,5]
    def __init__(self, depth):
        super(ConvNetSNopool, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 1 if i == 0 else 64
            outdim = 64
            # only first two layer has pooling and no padding
            B = ConvBlock(indim, outdim, pool=(
                i in [0, 1]), padding=0 if i in[0, 1] else 1)
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.final_feat_dim = [64, 5, 5]

    def forward(self, x):
        out = x[:, 0:1, :, :]  # only use the first dimension
        out = self.trunk(out)
        return out


class Attention(nn.Module):
    def __init__(self, inchannel, outchannel, way, shot):
        super(Attention, self).__init__()
        self.conv1 = nn.Conv2d(
            inchannel,
            outchannel,
            kernel_size=1,
            stride=1,
            padding=0)
        self.softmax = nn.Softmax(1)
        self.way = way
        self.shot = shot

    def predict(self, x):
        size = x.size()
        x = x.contiguous().view(self.way, -1, *size[1:])
        s = x[:, :self.shot, :, :, :]
        s = s.mean((1, 3, 4))
        q = x[:, self.shot:x.size(1), :, :, :]
        q = q.contiguous().view(self.way * (x.size(1)-self.shot), size[1], -1)
        return cosine(q, s)

    def extract(self, x):
        mask = self.conv1(x)
        input_size = mask.size()
        mask = mask.view(input_size[0], -1)
        mask = self.softmax(mask)
        mask = mask.reshape(input_size)
        return x.mul(mask)

    def forward(self, x):
        x = self.extract(x)

        return self.predict(x)


class ResNet(nn.Module):
    maml = False  # Default

    def __init__(
            self,
            block,
            list_of_num_layers,
            list_of_out_dims,
            flatten=True):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(ResNet, self).__init__()
        assert len(list_of_num_layers) == 4, 'Can have only four stages'
        if self.maml:
            conv1 = Conv2d_fw(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
            bn1 = BatchNorm2d_fw(64)
        else:
            conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                              bias=False)
            bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        for i in range(4):

            for j in range(list_of_num_layers[i]):
                half_res = (i >= 1) and (j == 0)
                B = block(indim, list_of_out_dims[i], half_res)
                trunk.append(B)
                indim = list_of_out_dims[i]

        if flatten:
            avgpool = nn.AvgPool2d(7)
            trunk.append(avgpool)
            trunk.append(Flatten())
            self.final_feat_dim = indim
        else:
            self.final_feat_dim = [indim, 7, 7]

        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        out = self.trunk(x)
        return out


class AttenNet(nn.Module):
    def __init__(
            self,
            block,
            list_of_num_layers,
            list_of_out_dims,
            way,
            shot):
        # list_of_num_layers specifies number of layers in each stage
        # list_of_out_dims specifies number of output channel for each stage
        super(AttenNet, self).__init__()
        assert len(list_of_num_layers) == 4, 'Can have only four stages'
        conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                          bias=False)
        bn1 = nn.BatchNorm2d(64)

        relu = nn.ReLU()
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        init_layer(conv1)
        init_layer(bn1)

        trunk = [conv1, bn1, relu, pool1]

        indim = 64
        self.atten1 = Attention(list_of_out_dims[0], 1, way, shot)
        self.atten2 = Attention(list_of_out_dims[1], 1, way, shot)
        self.atten3 = Attention(list_of_out_dims[2], 1, way, shot)
        self.atten4 = Attention(list_of_out_dims[3], 1, way, shot)
        self.convblock1 = []
        for j in range(list_of_num_layers[0]):
            half_res = False
            B = block(indim, list_of_out_dims[0], half_res)
            self.convblock1.append(B)
            indim = list_of_out_dims[0]
        self.convblock1 = nn.Sequential(*self.convblock1)

        self.convblock2 = []
        for j in range(list_of_num_layers[1]):
            half_res = (j == 0)
            B = block(indim, list_of_out_dims[1], half_res)
            self.convblock2.append(B)
            indim = list_of_out_dims[1]
        self.convblock2 = nn.Sequential(*self.convblock2)

        self.convblock3 = []
        for j in range(list_of_num_layers[2]):
            half_res = (j == 0)
            B = block(indim, list_of_out_dims[2], half_res)
            self.convblock3.append(B)
            indim = list_of_out_dims[2]
        self.convblock3 = nn.Sequential(*self.convblock3)

        self.convblock4 = []
        for j in range(list_of_num_layers[3]):
            half_res = False
            B = block(indim, list_of_out_dims[3], half_res)
            self.convblock4.append(B)
            indim = list_of_out_dims[3]
        self.convblock4 = nn.Sequential(*self.convblock4)

        self.final_feat_dim = [indim, 7, 7]
        self.coefficient = nn.Parameter(torch.tensor(
            [0.25, 0.25, 0.25, 0.25]), requires_grad=True)

        self.trunk = nn.Sequential(*trunk)

    def forward(self, x):
        out = self.trunk(x)
        out = self.convblock1(out)
        p1 = self.atten1(out)
        out = self.convblock2(out)
        p2 = self.atten2(out)
        out = self.convblock3(out)
        p3 = self.atten3(out)
        out = self.convblock4(out)
        p4 = self.atten4(out)
        p = torch.stack((p1, p2, p3, p4), dim=2)
        p = F.softmax(p.mul(self.coefficient).sum(2), dim=1)
        return p


def Conv4():
    return ConvNet(4)


def Conv6():
    return ConvNet(6)


def Conv4NP():
    return ConvNetNopool(4)


def Conv6NP():
    return ConvNetNopool(6)


def Conv4S():
    return ConvNetS(4)


def Conv4SNP():
    return ConvNetSNopool(4)


def AttenNet18(way=5, shot=5):
    return AttenNet(SimpleBlock, [2, 2, 2, 2], [64, 128, 256, 512], way, shot)


def AttenNet10(way=5, shot=5):
    return AttenNet(SimpleBlock, [1, 1, 1, 1], [64, 128, 256, 512], way, shot)


def ResNet10(flatten=True):
    return ResNet(SimpleBlock, [1, 1, 1, 1], [64, 128, 256, 512], flatten)


def ResNet18(flatten=True):
    return ResNet(SimpleBlock, [2, 2, 2, 2], [64, 128, 256, 512], flatten)


def ResNet34(flatten=True):
    return ResNet(SimpleBlock, [3, 4, 6, 3], [64, 128, 256, 512], flatten)


def ResNet50(flatten=True):
    return ResNet(
        BottleneckBlock, [
            3, 4, 6, 3], [
            256, 512, 1024, 2048], flatten)


def ResNet101(flatten=True):
    return ResNet(
        BottleneckBlock, [
            3, 4, 23, 3], [
            256, 512, 1024, 2048], flatten)
