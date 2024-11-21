import argparse
import torch
import torch.nn.functional as F
import yaml
from accelerate import Accelerator
from easydict import EasyDict
from src import utils
from src.data.loader_bmu import generate_data, get_transform, BMUDatasetPath
from src.model.bmunet import BMUNet

parser = argparse.ArgumentParser(description="arg parser")

parser.add_argument("--weight-path", type=str, default="./weight.bin")
parser.add_argument("--test-path", type=str, default="./test.csv")


args = parser.parse_args()


def get_dataset():
    test_list = generate_data(args.test_path)
    _, val_transform_mg, _, val_transform_us = get_transform(config)
    test_dataset = BMUDatasetPath(
        sample_list=test_list,
        transforms_mg=val_transform_mg,
        transforms_us=val_transform_us,
    )
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
    model = BMUNet()
    model_weight = torch.load(args.weight_path)
    model.load_state_dict(model_weight)

    model, test_loader = accelerator.prepare(model, test_loader)
    model.eval()

    with torch.no_grad():
        for i, (mg1, mg2, us3, us4, us5, us6, us7, us8, clinic_info, label, ori) in enumerate(
                test_loader
        ):
            data = (mg1, mg2, us3, us4, us5, us6, us7, us8)
            path = (
                ori["mg1_path"],
                ori["mg2_path"],
                ori["us3_path"],
                ori["us4_path"],
                ori["us5_path"],
                ori["us6_path"],
                ori["us7_path"],
                ori["us8_path"],
            )
            view = ori["view"]
            logist = model(mg1, mg2, us3, us4, us5, us6, us7, us8, clinic_info=clinic_info)
            scores = F.softmax(logist, dim=-1)
            pred = logist.argmax(dim=-1)
            scores_2 = torch.tensor(
                [torch.sum(scores[:, 0:2]), torch.sum(scores[:, 2:5], dim=1)]
            )
            # show_grad_cam(config, data, path, view, model, str(label.item()), str(pred.item()))

