import torch
import yaml
from easydict import EasyDict
from tqdm import tqdm

from src.data.loader_mg import get_dataloader


# Get dataset means and stds
def get_dataset_stats():
    config = EasyDict(
        yaml.load(
            open("../config.yml", "r", encoding="utf-8"),
            Loader=yaml.FullLoader,
        )
    )
    _, _, stats_data_loader = get_dataloader(config)
    means, stds = {0: [], 1: [], 2: []}, {0: [], 1: [], 2: []}
    indx = 1
    for batch in tqdm(stats_data_loader):
        tensor = batch[0]
        for channel in range(3):
            tensor_chan = tensor[:, channel]
            means[channel].append(torch.mean(tensor_chan.float()))
            stds[channel].append(torch.std(tensor_chan.float()))

        if indx % (len(stats_data_loader) // 20) == 0:
            _means = [torch.mean(torch.Tensor(means[channel])) for channel in range(3)]
            _stds = [torch.mean(torch.Tensor(stds[channel])) for channel in range(3)]
            print("for indx={}\t mean={}\t std={}\t".format(indx, _means, _stds))
        indx += 1
    means = [torch.mean(torch.Tensor(means[channel])) for channel in range(3)]
    stds = [torch.mean(torch.Tensor(stds[channel])) for channel in range(3)]
    return means, stds


if __name__ == "__main__":
    means, stds = get_dataset_stats()
    print(means, stds)
