from copy import copy, deepcopy
from ultralytics.models.yolo.segment import SegmentationTrainer
from ultralytics.nn.tasks import YOLOESegModel
from ultralytics.utils import RANK
from .train import YOLOETrainer, YOLOETrainerFromScratch, YOLOEVPTrainer
from .val import YOLOESegValidator


class YOLOESegTrainer(YOLOETrainer, SegmentationTrainer):

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = YOLOESegModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=min(self.data["nc"], 80),
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        self.loss_names = ("box", "seg", "cls", "dfl")
        return YOLOESegValidator(
            self.test_loader,
            save_dir=self.save_dir,
            args=copy(self.args),
            _callbacks=self.callbacks,
        )


class YOLOEPESegTrainer(SegmentationTrainer):

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = YOLOESegModel(
            cfg["yaml_file"] if isinstance(cfg, dict) else cfg,
            ch=self.data["channels"],
            nc=self.data["nc"],
            verbose=verbose and RANK == -1,
        )
        del model.model[-1].savpe
        assert (
            weights is not None
        ), "Pretrained weights must be provided for linear probing."
        if weights:
            model.load(weights)
        model.eval()
        names = list(self.data["names"].values())
        tpe = model.get_text_pe(names)
        model.set_classes(names, tpe)
        model.model[-1].fuse(model.pe)
        model.model[-1].cv3[0][2] = deepcopy(model.model[-1].cv3[0][2]).requires_grad_(
            True
        )
        model.model[-1].cv3[1][2] = deepcopy(model.model[-1].cv3[1][2]).requires_grad_(
            True
        )
        model.model[-1].cv3[2][2] = deepcopy(model.model[-1].cv3[2][2]).requires_grad_(
            True
        )
        if getattr(model.model[-1], "one2one_cv3", None) is not None:
            model.model[-1].one2one_cv3[0][2] = deepcopy(
                model.model[-1].cv3[0][2]
            ).requires_grad_(True)
            model.model[-1].one2one_cv3[1][2] = deepcopy(
                model.model[-1].cv3[1][2]
            ).requires_grad_(True)
            model.model[-1].one2one_cv3[2][2] = deepcopy(
                model.model[-1].cv3[2][2]
            ).requires_grad_(True)
        model.train()
        return model


class YOLOESegTrainerFromScratch(YOLOETrainerFromScratch, YOLOESegTrainer):
    pass


class YOLOESegVPTrainer(YOLOEVPTrainer, YOLOESegTrainerFromScratch):
    pass
