import torch
import torch.nn as nn

from .squeezenet import SqueezeNet
from .mobilenet import MobileNet, conv_bw

class ShadowSqueezeNet(SqueezeNet):
    def __init__(self) -> None:
        super().__init__(version="1_1", num_classes=10, dropout=0.5)
        self.layer1 = nn.Sequential(conv_bw(3, 32, 3, 1))  # mobilenet layer1
        self.midlayer = nn.ConvTranspose2d(
            32, 64, kernel_size=3, stride=2, padding=5, padding_mode="zeros"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.midlayer(x)
        x = self.features(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)

    def load_front_model(self, client_model):
        self.layer1.load_state_dict(client_model.layer1.state_dict())


class ShadowMobileNet(MobileNet):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )
        self.midlayer = nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        out = self.midlayer(x)
        out = self.feature(out)
        out = out.mean(3).mean(2)
        out = out.view(-1, 1024)
        out = self.classifer(out)
        return out

    def load_front_model(self, client_model):
        self.layer1.load_state_dict(client_model.layer1.state_dict())
