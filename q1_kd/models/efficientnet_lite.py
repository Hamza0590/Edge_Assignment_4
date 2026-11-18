"""
Student 3: EfficientNet-Lite0 implemented from scratch.

EfficientNet-Lite is the edge-friendly variant of EfficientNet-B0 released by
the TensorFlow team for mobile/edge accelerators. Differences from B0:
  * no Squeeze-and-Excitation blocks (poorly supported by edge accelerators),
  * ReLU6 instead of Swish (quantisation-friendly),
  * stem (32 ch) and head (1280 ch) are kept fixed when scaling.

Block = MBConv: 1x1 expand -> kxk depthwise -> 1x1 linear projection,
residual when stride 1 and in == out, with stochastic depth ("drop-connect")
on the residual branch during training.

CIFAR adaptation: stem stride 2 -> 1 and the first 24-channel stage stride
2 -> 1, so total down-sampling is 8x (32x32 -> 4x4 final map).
"""
import torch
import torch.nn as nn


def conv_bn_act(cin, cout, k=3, s=1, groups=1, act=True):
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
              nn.BatchNorm2d(cout, momentum=0.01, eps=1e-3)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class MBConv(nn.Module):
    def __init__(self, cin, cout, k, stride, expand, drop_path=0.0):
        super().__init__()
        hidden = cin * expand
        self.use_res = stride == 1 and cin == cout
        self.drop_path = drop_path
        layers = []
        if expand != 1:
            layers.append(conv_bn_act(cin, hidden, 1))
        layers += [conv_bn_act(hidden, hidden, k, stride, groups=hidden),
                   conv_bn_act(hidden, cout, 1, act=False)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if not self.use_res:
            return out
        if self.training and self.drop_path > 0:
            keep = 1.0 - self.drop_path
            mask = torch.empty(x.shape[0], 1, 1, 1, device=x.device, dtype=out.dtype).bernoulli_(keep)
            out = out / keep * mask
        return x + out


class EfficientNetLite0(nn.Module):
    # (expand, channels, repeats, stride, kernel) - B0 table; stage-2 stride changed 2 -> 1
    CFG = [(1, 16, 1, 1, 3), (6, 24, 2, 1, 3), (6, 40, 2, 2, 5), (6, 80, 3, 2, 3),
           (6, 112, 3, 1, 5), (6, 192, 4, 2, 5), (6, 320, 1, 1, 3)]
    feature_channels = 1280

    def __init__(self, num_classes=100, dropout=0.2, drop_path_rate=0.1):
        super().__init__()
        total = sum(c[2] for c in self.CFG)
        layers = [conv_bn_act(3, 32, 3, 1)]
        cin, idx = 32, 0
        for e, c, n, s, k in self.CFG:
            for i in range(n):
                layers.append(MBConv(cin, c, k, s if i == 0 else 1, e,
                                     drop_path=drop_path_rate * idx / total))
                cin, idx = c, idx + 1
        layers.append(conv_bn_act(cin, self.feature_channels, 1))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.feature_channels, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -1.0 / m.in_features ** 0.5, 1.0 / m.in_features ** 0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_features=False):
        feat = self.features(x)
        logits = self.classifier(torch.flatten(self.pool(feat), 1))
        return (logits, feat) if return_features else logits
