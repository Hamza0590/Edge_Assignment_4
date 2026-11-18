"""
Automated architecture generator: turns an architecture dict (see search_space.py)
into a runnable PyTorch nn.Module.

Block types
  conv   : kxk standard conv -> BN -> act
  dwsep  : depthwise-separable = kxk depthwise conv -> BN -> act -> 1x1 pointwise -> BN -> act
  mbconv : inverted residual  = 1x1 expand(x4) -> BN -> act -> kxk depthwise -> BN -> act
           -> 1x1 linear projection -> BN   (+ skip connection when shapes match)

Network = stem (3x3 conv) -> stages -> 1x1 conv head -> global avg pool -> dropout -> linear.
"""
import torch
import torch.nn as nn

ACT = {"relu": nn.ReLU, "relu6": nn.ReLU6, "hswish": nn.Hardswish, "silu": nn.SiLU}


def cba(cin, cout, k, s, act, groups=1):
    """conv-bn-(act)"""
    layers = [nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False), nn.BatchNorm2d(cout)]
    if act is not None:
        layers.append(ACT[act](inplace=True) if act != "hswish" else nn.Hardswish())
    return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k, s, act):
        super().__init__()
        self.body = cba(cin, cout, k, s, act)
        self.res = s == 1 and cin == cout

    def forward(self, x):
        return x + self.body(x) if self.res else self.body(x)


class DWSepBlock(nn.Module):
    def __init__(self, cin, cout, k, s, act):
        super().__init__()
        self.body = nn.Sequential(cba(cin, cin, k, s, act, groups=cin),   # depthwise
                                  cba(cin, cout, 1, 1, act))               # pointwise
        self.res = s == 1 and cin == cout

    def forward(self, x):
        return x + self.body(x) if self.res else self.body(x)


class MBConvBlock(nn.Module):
    def __init__(self, cin, cout, k, s, act, expand=4):
        super().__init__()
        h = cin * expand
        self.body = nn.Sequential(cba(cin, h, 1, 1, act), cba(h, h, k, s, act, groups=h),
                                  cba(h, cout, 1, 1, None))
        self.res = s == 1 and cin == cout

    def forward(self, x):
        return x + self.body(x) if self.res else self.body(x)


BLOCK = {"conv": ConvBlock, "dwsep": DWSepBlock, "mbconv": MBConvBlock}


class GeneratedNet(nn.Module):
    def __init__(self, arch, num_classes=10, head_channels=256, dropout=0.1):
        super().__init__()
        self.arch = arch
        act0 = arch["stages"][0]["activation"]
        layers = [cba(3, arch["stem_channels"], 3, 1, act0)]
        cin = arch["stem_channels"]
        for i, st in enumerate(arch["stages"]):
            for d in range(st["depth"]):
                stride = 2 if (i > 0 and d == 0) else 1
                layers.append(BLOCK[st["block"]](cin, st["channels"], st["kernel"], stride, st["activation"]))
                cin = st["channels"]
        layers.append(cba(cin, head_channels, 1, 1, arch["stages"][-1]["activation"]))
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_channels, num_classes))
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    @property
    def input_resolution(self):
        return self.arch["resolution"]

    def forward(self, x):
        return self.classifier(torch.flatten(self.pool(self.features(x)), 1))


def build_from_config(arch, num_classes=10):
    return GeneratedNet(arch, num_classes=num_classes)


def load_nas_checkpoint(path, map_location="cpu"):
    """Rebuild a GeneratedNet from a checkpoint that stores its architecture."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model = build_from_config(ck["meta"]["arch_config"], ck["meta"].get("num_classes", 10))
    model.load_state_dict(ck["state_dict"])
    return model, ck["meta"]
