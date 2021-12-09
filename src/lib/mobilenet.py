import torch.nn as nn

def conv_dw(inplane, outplane, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            inplane,
            inplane,
            kernel_size=3,
            groups=inplane,
            stride=stride,
            padding=1,
        ),
        nn.BatchNorm2d(inplane),
        nn.ReLU(),
        nn.Conv2d(inplane, outplane, kernel_size=1, groups=1, stride=1),
        nn.BatchNorm2d(outplane),
        nn.ReLU(),
    )


def conv_bw(inplane, outplane, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(
            inplane,
            outplane,
            kernel_size=kernel_size,
            groups=1,
            stride=stride,
            padding=1,
        ),
        nn.BatchNorm2d(outplane),
        nn.ReLU(),
    )


class MobileNet(nn.Module):
    def __init__(self, num_class=10):
        super(MobileNet, self).__init__()

        layers = []
        self.layer1 = nn.Sequential(conv_bw(3, 32, 3, 1))

        layers.append(conv_dw(32, 64, 1))
        layers.append(conv_dw(64, 128, 2))
        layers.append(conv_dw(128, 128, 1))
        layers.append(conv_dw(128, 256, 2))
        layers.append(conv_dw(256, 256, 1))
        layers.append(conv_dw(256, 512, 2))

        for i in range(5):
            layers.append(conv_dw(512, 512, 1))
        layers.append(conv_dw(512, 1024, 2))
        layers.append(conv_dw(1024, 1024, 1))

        self.classifer = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(1024, num_class)
        )
        self.feature = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer1(x)
        # --------------- cut
        out = self.feature(out)
        out = out.mean(3).mean(2)
        out = out.view(-1, 1024)
        out = self.classifer(out)
        return out


class ClientMobileNet(MobileNet):
    def forward(self, x):
        out = self.layer1(x)
        return out


class ServerMobileNet(MobileNet):
    def forward(self, x):
        x = self.feature(x)
        x = x.mean(3).mean(2)
        x = x.view(-1, 1024)
        out = self.classifer(x)
        return out


def SplitMobileNet(num_class=10):
    return ClientMobileNet(num_class), ServerMobileNet(num_class)
