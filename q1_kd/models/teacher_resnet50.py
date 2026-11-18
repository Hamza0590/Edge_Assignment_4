"""
Teacher: ResNet-50 adapted to 32x32 CIFAR images.

The ImageNet ResNet-50 stem (7x7/stride-2 conv + 3x3/stride-2 max-pool) shrinks a
32x32 image to 8x8 before the first residual stage, which throws away most of
the spatial information. Following the standard CIFAR adaptation we use a
3x3/stride-1 stem and drop the max-pool, so the four stages run at
32 -> 16 -> 8 -> 4 resolution.

The residual stages are initialised from torchvision's ImageNet weights
(transfer learning); the new stem and the 100-way classifier are trained from
scratch. `forward(x, return_features=True)` additionally returns the layer-4
feature map (2048 x 4 x 4) which is used for feature-based distillation.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class ResNet50Teacher(nn.Module):
    feature_channels = 2048

    def __init__(self, num_classes=100, pretrained=True):
        super().__init__()
        net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        net.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(net.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu)      # no max-pool
        self.layer1, self.layer2 = net.layer1, net.layer2
        self.layer3, self.layer4 = net.layer3, net.layer4
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.layer4(x)
        logits = self.fc(torch.flatten(self.pool(feat), 1))
        return (logits, feat) if return_features else logits
