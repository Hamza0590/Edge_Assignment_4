"""
Student 2: SqueezeNet (Iandola et al., 2016), v1.1 layout, implemented from scratch.

Fire module = 1x1 "squeeze" conv (few channels) followed by parallel 1x1 and
3x3 "expand" convs whose outputs are concatenated.

CIFAR adaptations (documented in decisions/):
  * 3x3 / stride-1 stem instead of 3x3 / stride-2 (32x32 inputs are tiny).
  * BatchNorm after every conv. The original SqueezeNet has no BN and is known
    to be hard to train from scratch (it often gets stuck at chance accuracy);
    BN makes training stable and is folded into the convs at ONNX export,
    so it costs nothing at inference time.
  * Three max-pools -> spatial path 32 -> 16 -> 8 -> 4, final map 512 x 4 x 4.
  * Classifier = dropout + 1x1 conv to num_classes + global average pool
    (the original SqueezeNet "conv classifier", no fully-connected layer).
"""
import torch
import torch.nn as nn


def conv_bn_relu(cin, cout, k, s=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class Fire(nn.Module):
    def __init__(self, cin, squeeze, expand1x1, expand3x3):
        super().__init__()
        self.squeeze = conv_bn_relu(cin, squeeze, 1)
        self.expand1 = conv_bn_relu(squeeze, expand1x1, 1)
        self.expand3 = conv_bn_relu(squeeze, expand3x3, 3)

    def forward(self, x):
        s = self.squeeze(x)
        return torch.cat([self.expand1(s), self.expand3(s)], dim=1)


class SqueezeNet(nn.Module):
    feature_channels = 512

    def __init__(self, num_classes=100, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn_relu(3, 64, 3, 1),                        # 32x32
            Fire(64, 16, 64, 64), Fire(128, 16, 64, 64),
            nn.MaxPool2d(2),                                  # 16x16
            Fire(128, 32, 128, 128), Fire(256, 32, 128, 128),
            nn.MaxPool2d(2),                                  # 8x8
            Fire(256, 48, 192, 192), Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            nn.MaxPool2d(2),                                  # 4x4
            Fire(512, 64, 256, 256),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Conv2d(512, num_classes, 1),
            nn.AdaptiveAvgPool2d(1))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.classifier[1].weight, 0, 0.01)

    def forward(self, x, return_features=False):
        feat = self.features(x)
        logits = torch.flatten(self.classifier(feat), 1)
        return (logits, feat) if return_features else logits
