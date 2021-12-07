import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.pass_by = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.pass_by = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.pass_by(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.pass_by = nn.Sequential()

        if stride != 1 or in_planes != planes * self.expansion:
            self.pass_by = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.pass_by(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_class=10, cutLvl=-1):
        super(ResNet, self).__init__()

        self.cutLvl = cutLvl

        self.inplanes = 64

        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(4, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_class)

    def _make_layer(self, block, plane, layer, stride=1):
        layers = []
        layers.append(block(self.inplanes, plane, stride))
        self.inplanes = plane * block.expansion
        for i in range(1, layer):
            layers.append(block(self.inplanes, plane))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ClientResNet(ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.cutLvl == 0:
            return x

        x = self.layer1(x)
        if self.cutLvl == 1:
            return x

        x = self.layer2(x)
        if self.cutLvl == 2:
            return x

        x = self.layer3(x)
        if self.cutLvl == 3:
            return x

        x = self.layer4(x)
        if self.cutLvl == 4:
            return x

        x = self.avgpool(x)

        return x


class ServerResNet(ResNet):
    def forward(self, x):

        # fall through
        if self.cutLvl <= 0:
            x = self.layer1(x)

        if self.cutLvl <= 1:
            x = self.layer2(x)

        if self.cutLvl <= 2:
            x = self.layer3(x)

        if self.cutLvl <= 3:
            x = self.layer4(x)

        if self.cutLvl <= 4:
            x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet18(num_class=10):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_class)


def ResNet34(num_class=10):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_class)


def ResNet50(num_class=10):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_class)
    return model


def SplitResNet18(num_class=10):
    return ClientResNet(BasicBlock, [2, 2, 2, 2], num_class), ServerResNet(
        BasicBlock, [2, 2, 2, 2], num_class
    )


def SplitResNet50(num_class=10):
    return ClientResNet(Bottleneck, [3, 4, 6, 3], num_class), ServerResNet(
        Bottleneck, [3, 4, 6, 3], num_class
    )
