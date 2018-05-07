import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import init
import pdb


class Vgg16(nn.Module):
    def __init__(self, pretrained=True):
        super(Vgg16, self).__init__()
        self.conv123 = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/2
            # conv2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/4
            # conv3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/8
        )
        self.conv4 = nn.Sequential(
            # conv4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),  # 1/16
        )
        self.conv5 = nn.Sequential(
            # conv5 features
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU()
        )
        if pretrained:
            vgg16 = torchvision.models.vgg16(pretrained=pretrained)
            L_vgg16 = list(vgg16.features)
            L_self = list(self.conv123)+list(self.conv4)+list(self.conv5)
            for l1, l2 in zip(L_vgg16, L_self):
                if (isinstance(l1, nn.Conv2d) and
                        isinstance(l2, nn.Conv2d)):
                    assert l1.weight.size() == l2.weight.size()
                    assert l1.bias.size() == l2.bias.size()
                    l2.weight.data = l1.weight.data
                    l2.bias.data = l1.bias.data

    def forward(self, x):
        x3 = self.conv123(x)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        return [x3, x4, x5]


def vgg(pretrained=True):
    model = Vgg16(pretrained)
    return model