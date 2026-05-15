import random
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (
    HOOKS,
    DistSamplerSeedHook,
    EpochBasedRunner,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
)
from mmcv.utils import build_from_cfg
from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.utils import get_root_logger


def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_detector(
    model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None
):
    logger = get_root_logger(cfg.log_level)
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if "imgs_per_gpu" in cfg.data:
        logger.warning(
            '"imgs_per_gpu" is deprecated in MMDet V2.0. Please use "samples_per_gpu" instead'
        )
        if "samples_per_gpu" in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and "samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"={cfg.data.imgs_per_gpu} is used in this experiments'
            )
        else:
            logger.warning(
                f'Automatically set "samples_per_gpu"="imgs_per_gpu"={cfg.data.imgs_per_gpu} in this experiments'
            )
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
        )
        for ds in dataset
    ]
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    optimizer = build_optimizer(model, cfg.optimizer)
    print(optimizer)
    runner = EpochBasedRunner(
        model, optimizer=optimizer, work_dir=cfg.work_dir, logger=logger, meta=meta
    )
    runner.timestamp = timestamp
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed
        )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )
    if distributed:
        runner.register_hook(DistSamplerSeedHook())
    if validate:
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
    if cfg.get("custom_hooks", None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(
            custom_hooks, list
        ), f"custom_hooks expect list type, but got {type(custom_hooks)}"
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(
                hook_cfg, dict
            ), f"Each item in custom_hooks expects dict type, but got {type(hook_cfg)}"
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop("priority", "NORMAL")
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def train_aet_detector(
    model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None
):
    logger = get_root_logger(cfg.log_level)
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if "imgs_per_gpu" in cfg.data:
        logger.warning(
            '"imgs_per_gpu" is deprecated in MMDet V2.0. Please use "samples_per_gpu" instead'
        )
        if "samples_per_gpu" in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and "samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"={cfg.data.imgs_per_gpu} is used in this experiments'
            )
        else:
            logger.warning(
                f'Automatically set "samples_per_gpu"="imgs_per_gpu"={cfg.data.imgs_per_gpu} in this experiments'
            )
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
        )
        for ds in dataset
    ]
    print(
        "backbone_number:",
        sum((param.numel() for param in model.backbone.parameters())),
    )
    print("neck_number:", sum((param.numel() for param in model.neck.parameters())))
    print(
        "neck_number:", sum((param.numel() for param in model.bbox_head.parameters()))
    )
    print("ref_head_number:", sum((param.numel() for param in model.ref.parameters())))
    param_group = []
    param_group += [{"params": model.backbone.parameters(), "lr": cfg.optimizer.lr[0]}]
    param_group += [{"params": model.neck.parameters(), "lr": cfg.optimizer.lr[0]}]
    param_group += [{"params": model.bbox_head.parameters(), "lr": cfg.optimizer.lr[0]}]
    param_group += [{"params": model.ref.parameters(), "lr": cfg.optimizer.lr[1]}]
    optimizer = torch.optim.SGD(
        param_group, cfg.optimizer.lr[0], momentum=0.9, weight_decay=0.0001
    )
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    runner = EpochBasedRunner(
        model, optimizer=optimizer, work_dir=cfg.work_dir, logger=logger, meta=meta
    )
    runner.timestamp = timestamp
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed
        )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )
    if distributed:
        runner.register_hook(DistSamplerSeedHook())
    if validate:
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
    if cfg.get("custom_hooks", None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(
            custom_hooks, list
        ), f"custom_hooks expect list type, but got {type(custom_hooks)}"
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(
                hook_cfg, dict
            ), f"Each item in custom_hooks expects dict type, but got {type(hook_cfg)}"
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop("priority", "NORMAL")
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)


def train_SR_detector(
    model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None
):
    logger = get_root_logger(cfg.log_level)
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    if "imgs_per_gpu" in cfg.data:
        logger.warning(
            '"imgs_per_gpu" is deprecated in MMDet V2.0. Please use "samples_per_gpu" instead'
        )
        if "samples_per_gpu" in cfg.data:
            logger.warning(
                f'Got "imgs_per_gpu"={cfg.data.imgs_per_gpu} and "samples_per_gpu"={cfg.data.samples_per_gpu}, "imgs_per_gpu"={cfg.data.imgs_per_gpu} is used in this experiments'
            )
        else:
            logger.warning(
                f'Automatically set "samples_per_gpu"="imgs_per_gpu"={cfg.data.imgs_per_gpu} in this experiments'
            )
        cfg.data.samples_per_gpu = cfg.data.imgs_per_gpu
    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
        )
        for ds in dataset
    ]
    print(
        "srbone_number:", sum((param.numel() for param in model.sr_bone.parameters()))
    )
    print(
        "backbone_number:",
        sum((param.numel() for param in model.backbone.parameters())),
    )
    print("neck_number:", sum((param.numel() for param in model.neck.parameters())))
    print(
        "head_number:", sum((param.numel() for param in model.bbox_head.parameters()))
    )
    param_group = []
    param_group += [{"params": model.sr_bone.parameters(), "lr": cfg.optimizer.lr[1]}]
    param_group += [{"params": model.backbone.parameters(), "lr": cfg.optimizer.lr[0]}]
    param_group += [{"params": model.neck.parameters(), "lr": cfg.optimizer.lr[0]}]
    param_group += [{"params": model.bbox_head.parameters(), "lr": cfg.optimizer.lr[0]}]
    optimizer = torch.optim.SGD(
        param_group, cfg.optimizer.lr[0], momentum=0.9, weight_decay=0.0001
    )
    if distributed:
        find_unused_parameters = cfg.get("find_unused_parameters", False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters,
        )
    else:
        model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    runner = EpochBasedRunner(
        model, optimizer=optimizer, work_dir=cfg.work_dir, logger=logger, meta=meta
    )
    runner.timestamp = timestamp
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed
        )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )
    if distributed:
        runner.register_hook(DistSamplerSeedHook())
    if validate:
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))
    if cfg.get("custom_hooks", None):
        custom_hooks = cfg.custom_hooks
        assert isinstance(
            custom_hooks, list
        ), f"custom_hooks expect list type, but got {type(custom_hooks)}"
        for hook_cfg in cfg.custom_hooks:
            assert isinstance(
                hook_cfg, dict
            ), f"Each item in custom_hooks expects dict type, but got {type(hook_cfg)}"
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop("priority", "NORMAL")
            hook = build_from_cfg(hook_cfg, HOOKS)
            runner.register_hook(hook, priority=priority)
    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow, cfg.total_epochs)
