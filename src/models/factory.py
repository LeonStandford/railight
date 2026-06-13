from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from .dsfd_vgg import build_net_vgg
from .dsfd_resnet import build_net_resnet
from .dai_net import build_net_dark


def build_net(phase, num_classes=2, model="vgg", **kw):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if model == "vgg":
        return build_net_vgg(phase, num_classes)
    elif model == "dark":
        return build_net_dark(phase, num_classes)
    elif model == "dark_sppf":
        return build_net_dark(phase, num_classes, enhance=True)
    elif "yolo" in model:
        from .idayolo import build_idayolo

        scale = kw.get("scale") or (
            model[6:] if model.startswith("yolo26") and len(model) > 6 else "n"
        )
        return build_idayolo(
            phase, num_classes, scale=scale, weights=kw.get("weights", "auto")
        )
    else:
        return build_net_resnet(phase, num_classes, model)


def basenet_factory(model="vgg"):
    if model in ("vgg", "dark", "dark_sppf"):
        basenet = "vgg16_reducedfc.pth"
    elif "yolo" in model:
        basenet = ""
    elif "resnet" in model:
        basenet = "{}.pth".format(model)
    else:
        basenet = ""
    return basenet
