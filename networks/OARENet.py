from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

from networks.decoder import build_decoder
from networks.intersection import build_erase


class OARENet(nn.Module):
    def __init__(
        self,
        pretrained_backbone: bool = True,
        erase_channels: int = 2,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained_backbone else None
        backbone = resnet50(weights=weights)

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.decoder = build_decoder(filters=[256, 512, 1024, 2048])
        self.erase = build_erase(in_channel=64, erase_channel=erase_channels)
        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(64 + erase_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        e1 = self.layer1(x)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)

        decoded = self.decoder(e1, e2, e3, e4)
        erase_features = self.erase(decoded)
        upsampled = self.up(decoded)
        logits = self.head(torch.cat([erase_features, upsampled], dim=1))
        return logits


GAMSNet_OAM = OARENet

