#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

# general
import pandas as pd
import math
import shutil
import numpy as np
import sys
import logging
import argparse
from pathlib import Path, PurePath
import os
import glob
import json
import re

# others
import wandb
from PIL import Image
from torchinfo import summary
from qqdm import qqdm as tqdm
from pytorchcv.model_provider import get_model as ptcv_get_model
from torchattacks import TIFGSM

from attack import (
    AdvDataset,
    Attacker
)

logger = logging.getLogger(__name__)

cuclear = torch.cuda.empty_cache


class SoftLabelDataset(AdvDataset):
    def __init__(self, data_dir, transform, lprobs):
        super().__init__(data_dir, transform)
        self.lprobs = lprobs

    def __getitem__(self, idx):
        x, y = super().__getitem__(idx)
        return (x, y, self.lprobs[idx])


class Calculator(Attacker):
    def __init__(self, args):
        super().__init__(args)

        self.adv_set = SoftLabelDataset(
            args.datadir,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.cifar_100_mean, self.cifar_100_std)
            ]),
            lprobs=self.load_target_lprobs(args.csv_path)  # add
        )
        self.adv_loader = DataLoader(self.adv_set, batch_size=self.batch_size, shuffle=False)
        # self.loss_fn = nn.KLDivLoss(reduction='sum', log_target=True)

    def load_target_lprobs(self, path):
        df = pd.read_csv(path, skiprows=1)

        data = {}
        for index, row in df.iterrows():
            img = row["name"].strip()
            logits = torch.zeros((100,))  # cifar 100
            for i in range(1, 6):
                cls = int(re.search(r"(?P<x>[0-9]+)-.+", row[f"top{i} class"]).group("x"))
                val = float(row[f"top{i} logit"])
                assert 0 <= cls <= 99
                logits[cls] = val
                # print(logits.sum())
            if self.args.logit:
                data[img] = logits
            else:
                remainder = (1. - logits.sum()) / (logits == 0).sum()
                data[img] = torch.where(logits > 0, logits, remainder)
                assert torch.isclose(data[img].sum(), torch.ones((1,)))

        ordered_data = []
        for im in self.adv_set.images:
            ordered_data.append(data[PurePath(im).name])
        ordered_data = torch.stack(ordered_data, 0)  # N, 100

        # res = [int(n.split('_')[0]) == int(p.split('-')[0]) for n, p in zip(df["name"], df["top1 class"])]
        # logger.info(f"top-1: {sum(res) / len(res)}")

        return ordered_data.log_softmax(-1) if self.args.logit else ordered_data.log()

    @torch.no_grad()
    def calc_kl_div(self, model):
        device = self.device
        model.eval()
        n_data = len(self.adv_loader.dataset)
        kl_div, rev_kl_div = 0.0, 0.0
        for i, (x, y, z) in enumerate(self.adv_loader):
            cuclear()
            x, z = x.to(device), z.to(device)
            lprob = model(x).log_softmax(-1)
            kl_div += F.kl_div(
                lprob,
                z,
                reduction='sum',
                log_target=True
            ).item()
            rev_kl_div += F.kl_div(
                z,
                lprob,
                reduction='sum',
                log_target=True
            ).item()
        return kl_div / n_data, rev_kl_div / n_data

    def solve(self, ):
        data = {}
        for modelname in self.args.model.split(","):
            model = self.build_model(modelname)
            kl, rev_kl = self.calc_kl_div(model)
            logger.info("model: {} kl: {:.03f}, rev_kl: {:.03f}".format(modelname, kl, rev_kl))
            data[modelname] = {"kl": kl, "rev_kl": rev_kl}
        df = pd.DataFrame.from_dict(data).T
        df.to_csv(self.args.result_path)

    @staticmethod
    def add_args(parser):
        super(Calculator, Calculator).add_args(parser)
        parser.add_argument("--csv-path", default="./benign_result.csv")
        parser.add_argument("--result-path", default="./kld_result.csv")
        parser.add_argument("--logit", action="store_true")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    Calculator.add_args(parser)

    args = parser.parse_args()
    Calculator(args).solve()
