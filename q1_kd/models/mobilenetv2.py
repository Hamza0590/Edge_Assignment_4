"""
Student 1: MobileNetV2 (Sandler et al., 2018) implemented from scratch.

Building block = inverted residual:
    1x1 expand conv (t x channels) -> 3x3 depthwise conv -> 1x1 linear projection
with a residual connection when stride == 1 and in/out channels match.

CIFAR adaptation: the ImageNet network downsamples 32x; on 32x32 inputs that
would leave a 1x1 map. We use stride 1 in the stem and in the second stage
(the first "24-channel" stage), giving a total down-sampling of 8x
(32 -> 4x4 final feature map), the usual choice for CIFAR.
"""
import torch
import torch.nn as nn


def conv_bn_act(cin, cout, k=3, s=1, groups=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    def __init__(self, cin, cout, stride, expand):
        super().__init__()
        hidden = cin * expand
        self.use_res = stride == 1 and cin == cout
        layers = []
        if expand != 1:
            layers.append(conv_bn_act(cin, hidden, k=1))                     # expansion
        layers += [conv_bn_act(hidden, hidden, k=3, s=stride, groups=hidden),  # depthwise
                   conv_bn_act(hidden, cout, k=1, act=False)]                # linear bottleneck
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x) if self.use_res else self.block(x)


class MobileNetV2(nn.Module):
    # (expansion t, output channels c, repeats n, stride s) - stride of stage 2 changed 2 -> 1
    CFG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2),
           (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]
    feature_channels = 1280

    def __init__(self, num_classes=100, width_mult=1.0, dropout=0.2):
        super().__init__()
        cin = int(32 * width_mult)
        layers = [conv_bn_act(3, cin, k=3, s=1)]                 # stem stride 1 for CIFAR
        for t, c, n, s in self.CFG:
            cout = int(c * width_mult)
            for i in range(n):
                layers.append(InvertedResidual(cin, cout, s if i == 0 else 1, t))
                cin = cout
        layers.append(conv_bn_act(cin, self.feature_channels, k=1))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_channels, num_classes))
        _init_weights(self)

    def forward(self, x, return_features=False):
        feat = self.features(x)
        logits = self.classifier(torch.flatten(self.pool(feat), 1))
        return (logits, feat) if return_features else logits


def _init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.zeros_(m.bias)
