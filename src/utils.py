import os
import sys
from collections import Counter
import numpy as np
import torch
from torch import nn


class MetricSaver(nn.Module):
    def __init__(self):
        super().__init__()
        self.best_acc = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.current_acc = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.epoch = nn.Parameter(torch.zeros(1), requires_grad=False)


def same_seeds(seed):
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


class Logger(object):
    def __init__(self, logdir: str):
        self.console = sys.stdout
        if logdir is not None:
            os.makedirs(logdir)
            self.log_file = open(logdir + "/log.txt", "w")
        else:
            self.log_file = None
        sys.stdout = self
        sys.stderr = self

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.log_file is not None:
            self.log_file.write(msg)

    def flush(self):
        self.console.flush()
        if self.log_file is not None:
            self.log_file.flush()
            os.fsync(self.log_file.fileno())

    def close(self):
        self.console.close()
        if self.log_file is not None:
            self.log_file.close()


def get_sampler(train_dataset):
    label_dist = [d["label"] for d in train_dataset.sample_list]
    label_counts = Counter(label_dist)
    weight_per_label = 1.0 / len(label_counts)
    label_weights = {
        label: weight_per_label / count for label, count in label_counts.items()
    }
    weights = [label_weights[label] for label in label_dist]

    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )
    return sampler
