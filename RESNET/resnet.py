import torch.nn as nn

class BasicBlock(nn.Module):
    """
    Basic block for resnet18 or resnet34
    expansion = 1
    structure: two 3x3 convolutions
    """

    expansion = 1

    def __init__(self, in_features, out_features, stride=1, downsample=None):

        super(BasicBlock, self).__init__()
        self.basicblock = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=3, stride=stride,
                    padding=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_features)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.basicblock(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BottleNeck(nn.Module):
    """
    Basic block for resnet50 and resnet with more layers
    expansion = 4
    structure:  1x1 convolution + 3x3 convolution + 1x1 convolution
    """

    expansion = 4

    def __init__(self, in_features, out_features, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=1, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_features, out_channels=out_features, kernel_size=3, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_features, out_channels=out_features * self.expansion, kernel_size=1, stride=stride,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_features * self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64  # default value
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # block
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])

def ResNet50():
    return ResNet(BottleNeck, [3, 4, 6, 3])

def ResNet101():
    return ResNet(BottleNeck, [3, 4, 23, 3])

def ResNet152():
    return ResNet(BottleNeck, [3, 8, 36, 3])
