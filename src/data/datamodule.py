from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
import torch
import torch.utils.data as data
from data.config import cfg as _global_cfg
from data.source_domain import SourceDomainDetection, detection_collate
from data.target_domain import TargetUnlabeledDataset

__all__ = ["DataModule", "DataBundle"]


@dataclass
class DataBundle:
    train_dataset: SourceDomainDetection
    train_loader: data.DataLoader
    val_dataset: SourceDomainDetection
    val_loader: data.DataLoader
    target_dataset: Optional[TargetUnlabeledDataset]
    target_loader: Optional[data.DataLoader]


class DataModule:

    def __init__(
        self,
        *,
        train_file: str,
        val_file: str,
        batch_size: int,
        num_workers: int = 0,
        target_folder: str = "",
        input_size: Optional[int] = None,
    ) -> None:
        self.train_file = train_file
        self.val_file = val_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_folder = target_folder
        self.input_size = int(input_size or _global_cfg.INPUT_SIZE)

    def build(self) -> DataBundle:
        train_ds = SourceDomainDetection(self.train_file, mode="train")
        train_loader = data.DataLoader(
            train_ds,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=detection_collate,
            sampler=torch.utils.data.distributed.DistributedSampler(
                train_ds, shuffle=True
            ),
            pin_memory=True,
        )
        val_ds = SourceDomainDetection(self.val_file, mode="val")
        val_loader = data.DataLoader(
            val_ds,
            self.batch_size,
            num_workers=0,
            collate_fn=detection_collate,
            sampler=torch.utils.data.distributed.DistributedSampler(
                val_ds, shuffle=False
            ),
            pin_memory=True,
        )
        target_ds: Optional[TargetUnlabeledDataset] = None
        target_loader: Optional[data.DataLoader] = None
        if self.target_folder and os.path.isdir(self.target_folder):
            target_ds = TargetUnlabeledDataset(self.target_folder, size=self.input_size)
            if len(target_ds) > 0:
                target_loader = data.DataLoader(
                    target_ds,
                    self.batch_size,
                    num_workers=self.num_workers,
                    sampler=data.RandomSampler(
                        target_ds, replacement=True, num_samples=int(1000000000000.0)
                    ),
                    pin_memory=True,
                    drop_last=True,
                )
        return DataBundle(
            train_dataset=train_ds,
            train_loader=train_loader,
            val_dataset=val_ds,
            val_loader=val_loader,
            target_dataset=target_ds,
            target_loader=target_loader,
        )
