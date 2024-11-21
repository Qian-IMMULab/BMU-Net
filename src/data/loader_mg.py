from typing import List, Any, Tuple, Dict
import pandas as pd
import torch
import torchvision.datasets
from PIL import Image
from easydict import EasyDict
from torchvision.transforms import Compose

from src.utils import get_sampler


class MGDatasetPath(torch.utils.data.Dataset):
    def __init__(
            self, sample_list: List[dict], transforms_mg: torchvision.transforms.Compose
    ) -> None:
        super(MGDatasetPath, self).__init__()

        self.sample_list = sample_list
        self.transforms_mg = transforms_mg
        self.loader = torchvision.datasets.folder.default_loader

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index: int) -> tuple[Any, Any, dict[Any, Any]]:
        # mg1: CC mg2: MLO
        sample = self.sample_list[index]
        mg_paths = [sample[f"mg{i}_path"] for i in range(1, 3)]
        label = self.sample_list[index]["label"]

        mg_images = [
            (Image.open(path) if path != 'N' else Image.new("I", (1280, 2294), 0))
            for path in mg_paths
        ]

        mg_tensors = [self.transforms_mg(img) for img in mg_images]
        # Original paths for return
        ori = {key: value for key, value in sample.items() if key.endswith('_path') or key == 'view'}
        return *mg_tensors, label, ori


class MGDataset(MGDatasetPath):
    def __getitem__(self, index: int) -> tuple[Any, Any, Any]:
        mg1, mg2, label, ori = super().__getitem__(index)
        return mg1, mg2, label


def get_dataloader(config: EasyDict) -> tuple[Any, Any, Any]:
    train_list = generate_data(config.mg.train_data_dir)
    val_list = generate_data(config.mg.val_data_dir)
    # For getting dataset means and stds
    stats_list = generate_data(config.mg.stats_data_dir)

    train_transform_mg, val_transform_mg, stats_transform_mg = get_transform(config)

    train_dataset = MGDataset(sample_list=train_list, transforms_mg=train_transform_mg)
    val_dataset = MGDataset(sample_list=val_list, transforms_mg=val_transform_mg)
    stats_dataset = MGDataset(sample_list=stats_list, transforms_mg=stats_transform_mg)

    # Extraction of training samples by weight
    sampler = get_sampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=config.mg.batch_size,
        num_workers=config.trainer.num_workers,
        shuffle=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.mg.batch_size,
        num_workers=config.trainer.num_workers,
        shuffle=False,
    )
    stats_data_loader = torch.utils.data.DataLoader(
        stats_dataset,
        batch_size=config.mg.batch_size,
        num_workers=config.trainer.num_workers,
        shuffle=True,
    )
    return train_loader, val_loader, stats_data_loader


def generate_data(data_dir):
    sample_list = []
    df = pd.read_csv(data_dir, skiprows=0)
    for index, row in df.iterrows():
        data = {
            "patient_id": row["patient_id"],
            "exam_id": row["exam_id"],
            "mg1_path": row["mg1_path"],
            "mg2_path": row["mg2_path"],
            "label": row["label"],
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


def get_transform(config: EasyDict) -> tuple[Compose, Compose, Compose]:
    channel_means = [config.mg.img_mean]
    channel_stds = [config.mg.img_std]
    train_transform_mg = torchvision.transforms.Compose(
        [

            torchvision.transforms.Resize(
                size=(config.mg.image_h, config.mg.image_w)
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
                size=(config.mg.image_h, config.mg.image_w)
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(force_num_chan),
            torchvision.transforms.Normalize(
                torch.Tensor(channel_means), torch.Tensor(channel_stds)
            ),
        ]
    )

    stats_transform_mg = torchvision.transforms.Compose(
        [

            torchvision.transforms.Resize(
                size=(config.mg.image_h, config.mg.image_w)
            ),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Lambda(force_num_chan),
        ]
    )
    return train_transform_mg, val_transform_mg, stats_transform_mg
