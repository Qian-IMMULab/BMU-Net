import argparse
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from easydict import EasyDict

from src import utils
from src.data.loader_mg import generate_data, get_transform, MGDatasetPath
from src.model.bmunet import MGModuleCAM

parser = argparse.ArgumentParser(description="arg parser")

parser.add_argument("--weight-path", type=str, default="./weight.bin")
parser.add_argument("--test-path", type=str, default="./test.csv")

args = parser.parse_args()


def get_dataset():
    test_list = generate_data(args.test_path)
    _, val_transform_mg, _ = get_transform(config)
    test_dataset = MGDatasetPath(sample_list=test_list, transforms_mg=val_transform_mg)
    loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, num_workers=1, shuffle=False
    )
    return loader


if __name__ == "__main__":

    # Load config
    config = EasyDict(
        yaml.load(open("config.yml", "r", encoding="utf-8"), Loader=yaml.FullLoader)
    )
    accelerator = Accelerator()
    utils.same_seeds(42)

    # Load test data
    test_loader = get_dataset()

    # Load model weight
    model = MGModuleCAM()
    model_weight = torch.load(args.weight_path)
    model.load_state_dict(model_weight)

    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()

    for i, (mg1, mg2, label, ori) in enumerate(test_loader):
        logist = model(mg1, mg2)
        scores = F.softmax(logist, dim=1)
        pred = logist.argmax(dim=-1)
        scores_2 = torch.tensor(
            [torch.sum(scores[:, 0:2]), torch.sum(scores[:, 2:5], dim=1)]
        )
