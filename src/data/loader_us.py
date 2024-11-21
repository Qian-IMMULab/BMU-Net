from typing import Tuple, List, Any
import pandas as pd
import torch
import torchvision.datasets
from PIL import Image
from easydict import EasyDict
from torchvision.transforms import Compose

from src.utils import get_sampler


class USDatasetPath(torch.utils.data.Dataset):
    def __init__(
        self, sample_list: List[dict], transforms_us: torchvision.transforms.Compose
    ) -> None:
        super(USDatasetPath, self).__init__()

        self.sample_list = sample_list
        self.transforms_us = transforms_us
        self.loader = torchvision.datasets.folder.default_loader

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(
        self, index: int
    ) -> tuple[Any, Any, dict[Any, Any]]:
        # us3 and us4 are a modality, and so on
        sample = self.sample_list[index]
        us_paths = [sample[f"us{i}_path"] for i in range(3, 9)]
        label = self.sample_list[index]["label"]
        us_images = [
            (self.loader(path) if path != 'N' else Image.new("RGB", (224, 224), (0, 0, 0)))
            for path in us_paths
        ]

        us_tensors = [self.transforms_us(img) for img in us_images]
        # Original paths for return
        ori = {key: value for key, value in sample.items() if key.endswith('_path') or key == 'view'}

        return *us_tensors, label, ori


class USDataset(USDatasetPath):
    def __getitem__(self, index: int) -> tuple[Any, Any, Any, Any, Any, Any, Any]:
        us3, us4, us5, us6, us7, us8, label, ori = super().__getitem__(index)
        return us3, us4, us5, us6, us7, us8, label


def get_dataloader(
    config: EasyDict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_list = generate_data(config.us.train_data_dir)
    val_list = generate_data(config.us.val_data_dir)

    train_transform_us, val_transform_us = get_transform(config)

    train_dataset = USDataset(sample_list=train_list, transforms_us=train_transform_us)
    val_dataset = USDataset(sample_list=val_list, transforms_us=val_transform_us)
    sampler = get_sampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=config.us.batch_size,
        num_workers=config.trainer.num_workers,
        shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.us.batch_size,
        num_workers=config.trainer.num_workers,
        shuffle=False,
    )
    return train_loader, val_loader


def generate_data(data_dir):
    sample_list = []
    df = pd.read_csv(data_dir, skiprows=0)
    for index, row in df.iterrows():
        data = {
            "patient_id": row["patient_id"],
            "exam_id": row["exam_id"],
            "us3_path": row["us3_path"],
            "us4_path": row["us4_path"],
            "us5_path": row["us5_path"],
            "us6_path": row["us6_path"],
            "us7_path": row["us7_path"],
            "us8_path": row["us8_path"],
            "label": row["label"],
        }
        sample_list.append(data)
    return sample_list


def get_transform(config: EasyDict) -> tuple[Compose, Compose]:
    train_transform_us = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                size=(config.us.image_size, config.us.image_size)
            ),
            torchvision.transforms.RandomAffine(0, (0.1, 0)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomRotation(degrees=10),
            torchvision.transforms.GaussianBlur(kernel_size=3),
            torchvision.transforms.ToTensor(),
        ]
    )

    val_transform_us = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                size=(config.us.image_size, config.us.image_size)
            ),
            torchvision.transforms.ToTensor(),
        ]
    )
    return train_transform_us, val_transform_us
