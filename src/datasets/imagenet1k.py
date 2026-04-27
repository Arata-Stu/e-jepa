from __future__ import annotations

from pathlib import Path

import torch
import torchvision


class ImageNet(torchvision.datasets.ImageFolder):
    def __init__(self, root: str | Path, transform=None, train: bool = True):
        suffix = "train" if train else "val"
        data_path = Path(root).expanduser() / suffix
        super().__init__(root=str(data_path), transform=transform)


class ImageNetSubset:
    def __init__(self, dataset: ImageNet, subset_file: str | Path):
        self.dataset = dataset
        self.subset_file = Path(subset_file).expanduser()
        self.samples = self._read_subset_samples()

    def _read_subset_samples(self):
        class_to_idx = self.dataset.class_to_idx
        rows = []
        with self.subset_file.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                class_name = line.split("_")[0]
                target = class_to_idx[class_name]
                img_path = self.dataset.root + f"/{class_name}/{line}"
                rows.append((img_path, target))
        return rows

    @property
    def classes(self):
        return self.dataset.classes

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.dataset.loader(path)
        if self.dataset.transform is not None:
            img = self.dataset.transform(img)
        if self.dataset.target_transform is not None:
            target = self.dataset.target_transform(target)
        return img, target


def make_imagenet1k(
    transform,
    batch_size,
    *,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    training=True,
    drop_last=True,
    persistent_workers=False,
    subset_file=None,
):
    dataset = ImageNet(root=root_path, transform=transform, train=training)
    if subset_file is not None:
        dataset = ImageNetSubset(dataset, subset_file=subset_file)

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=training,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        pin_memory=pin_mem,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0) and persistent_workers,
    )
    return dataset, data_loader, sampler
