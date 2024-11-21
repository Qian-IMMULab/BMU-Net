from typing import Tuple, List, Any
import pandas as pd
import torch
import torchvision.datasets
from PIL import Image
from easydict import EasyDict
from torchvision.transforms import Compose

from src.utils import get_sampler


class BMUDatasetPath(torch.utils.data.Dataset):
    def __init__(
            self,
            sample_list: List[dict],
            transforms_mg: torchvision.transforms.Compose,
            transforms_us: torchvision.transforms.Compose,
    ) -> None:
        super(BMUDatasetPath, self).__init__()

        self.sample_list = sample_list
        self.transforms_mg = transforms_mg
        self.transforms_us = transforms_us
        self.loader = torchvision.datasets.folder.default_loader

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(
            self, index: int
    ) -> tuple[Any, Any, Any, Any, dict[Any, Any]]:
        sample = self.sample_list[index]
        mg_paths = [sample[f"mg{i}_path"] for i in range(1, 3)]
        us_paths = [sample[f"us{i}_path"] for i in range(3, 9)]
        label = sample["label"]
        clinic_info = sample["clinic_info"]

        mg_images = [
            (Image.open(path) if path != 'N' else Image.new("I", (1280, 2294), 0))
            for path in mg_paths
        ]

        mg_tensors = [self.transforms_mg(img) for img in mg_images]

        us_images = [
            (self.loader(path) if path != 'N' else Image.new("RGB", (224, 224), (0, 0, 0)))
            for path in us_paths
        ]

        us_tensors = [self.transforms_us(img) for img in us_images]

        # Original paths for return
        ori = {key: value for key, value in sample.items() if key.endswith('_path') or key == 'view'}

        return *mg_tensors, *us_tensors, clinic_info, label, ori


class BMUDataset(BMUDatasetPath):
    def __getitem__(
            self, index: int
    ) -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any, Any]:
        mg1, mg2, us3, us4, us5, us6, us7, us8, clinic_info, label, ori = super().__getitem__(index)
        return mg1, mg2, us3, us4, us5, us6, us7, us8, clinic_info, label


def get_dataloader(
        config: EasyDict,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    if config.bmu.split == "static_split":
        train_list = generate_data(config.bmu.train_data_dir)
        val_list = generate_data(config.bmu.val_data_dir)
    elif config.bmu.split == "random_split":
        sample_list = generate_data(config.bmu.train_data_dir)
        train_size = int(len(sample_list) * config.bmu.train_ratio)
        train_list, val_list = torch.utils.data.random_split(
            sample_list, [train_size, len(sample_list) - train_size]
        )

    (
        train_transform_mg,
        val_transform_mg,
        train_transform_us,
        val_transform_us,
    ) = get_transform(config)

    train_dataset = BMUDataset(
        sample_list=train_list,
        transforms_mg=train_transform_mg,
        transforms_us=train_transform_us,
    )
    val_dataset = BMUDataset(
        sample_list=val_list,
        transforms_mg=val_transform_mg,
        transforms_us=val_transform_us,
    )

    # Extraction of training samples by weight
    sampler = get_sampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=config.bmu.batch_size,
        num_workers=config.trainer.num_workers,
        shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.bmu.batch_size,
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
            "mg1_path": row["mg1_path"],
            "mg2_path": row["mg2_path"],
            "us3_path": row["us3_path"],
            "us4_path": row["us4_path"],
            "us5_path": row["us5_path"],
            "us6_path": row["us6_path"],
            "us7_path": row["us7_path"],
            "us8_path": row["us8_path"],
            "label": row["label"],
            "clinic_info": torch.Tensor(
                [
                    row["clinic_info1"],
                    row["clinic_info2"],
                    row["clinic_info3"],
                    row["clinic_info4"],
                    row["clinic_info5"],
                    row["clinic_info6"],
                    row["clinic_info7"],
                    row["clinic_info8"],
                    row["clinic_info9"],
                    row["clinic_info10"],
                ]
            ),
            "view": row["view"],

        }
        sample_list.append(data)
    return sample_list


# Image channel to 3
def force_num_chan(data_tensor):
    data_tensor = data_tensor.float()
    existing_chan = data_tensor.size()[0]
    if not existing_chan == 3:
        return data_tensor.expand(3, *data_tensor.size()[1:])
    return data_tensor


def get_transform(config: EasyDict) -> tuple[Compose, Compose, Compose, Compose]:
    channel_means = [config.bmu.img_mean]
    channel_stds = [config.bmu.img_std]
    train_transform_mg = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                size=(config.bmu.mg_image_h, config.bmu.mg_image_w)
            ),
            torchvision.transforms.RandomAffine(0, (0.1, 0)),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomRotation(degrees=(-20, +20)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(force_num_chan),
            torchvision.transforms.Normalize(
                torch.Tensor(channel_means), torch.Tensor(channel_stds)
            ),
        ]
    )

    val_transform_mg = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                size=(config.bmu.mg_image_h, config.bmu.mg_image_w)
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(force_num_chan),
            torchvision.transforms.Normalize(
                torch.Tensor(channel_means), torch.Tensor(channel_stds)
            ),
        ]
    )

    train_transform_us = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(
                size=(config.bmu.us_image_size, config.bmu.us_image_size)
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
                size=(config.bmu.us_image_size, config.bmu.us_image_size)
            ),
            torchvision.transforms.ToTensor(),
        ]
    )
    return train_transform_mg, val_transform_mg, train_transform_us, val_transform_us
