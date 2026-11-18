"""Model registry so every script can build any network by name."""
from .teacher_resnet50 import ResNet50Teacher
from .mobilenetv2 import MobileNetV2
from .squeezenet import SqueezeNet
from .efficientnet_lite import EfficientNetLite0

STUDENTS = {
    "mobilenetv2": MobileNetV2,
    "squeezenet": SqueezeNet,
    "efficientnet_lite0": EfficientNetLite0,
}


def build_model(name, num_classes=100, **kw):
    if name == "resnet50":
        return ResNet50Teacher(num_classes=num_classes, **kw)
    return STUDENTS[name](num_classes=num_classes, **kw)
