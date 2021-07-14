import torch.nn as nn
from se_layers import SELayer
from resnet import ResNet


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    """
    Basic block for resnet18 or resnet34
    expansion = 1
    structure: two 3x3 convolutions
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, *, reduction=16):
        super(SEBasicBlock, self).__init__()

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class SEBottleneck(nn.Module):
    """
    Basic block for resnet50 and resnet with more layers
    expansion = 4
    structure:  1x1 convolution + 3x3 convolution + 1x1 convolution
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def SEResNet18(n_classes):
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=n_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

def SEResNet34(n_classes):
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=n_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def SEResNet50(n_classes):
    model = ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=n_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

def SEResNet101(n_classes):
    model = ResNet(SEBasicBlock, [3, 4, 23, 3], num_classes=n_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

def SEResNet152(n_classes):
    model = ResNet(SEBasicBlock, [3, 8, 36, 3], num_classes=n_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model
