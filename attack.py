import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

# general
import shutil
import numpy as np
import sys
import logging
import argparse
from pathlib import Path
import os
import glob
import json

# others
from PIL import Image
from torchinfo import summary
from qqdm import qqdm as tqdm
from pytorchcv.model_provider import get_model as ptcv_get_model

# local
import models
from models.linbp_utils import (
    linbp_forw_resnet50,
    linbp_backw_resnet50,
)
logger = logging.getLogger(__name__)

cuclear = torch.cuda.empty_cache
@torch.no_grad()
def clamp(x_adv, x, epsilon, margin=1e-6):
    tmp = x + (epsilon - margin)
    x_adv = torch.where(x_adv > tmp, tmp, x_adv)
    tmp = x - (epsilon - margin)
    x_adv = torch.where(x_adv < tmp, tmp, x_adv)
    return x_adv

class AdvDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.images = []
        self.labels = []
        self.names = []
        '''
        data_dir
        ├── class_dir
        │   ├── class1.png
        │   ├── ...
        │   ├── class20.png
        '''
        for imagename in sorted(glob.glob(f'{data_dir}/*')):
            basename = os.path.basename(imagename)
            cls, name = basename.split('.')[0].split('_')
            self.images += [imagename]
            self.labels += [int(cls)]
            self.names += [basename]
        self.transform = transform
    def __getitem__(self, idx):
        image = self.transform(Image.open(self.images[idx]))
        label = self.labels[idx]
        return image, label
    def __getname__(self):
        return self.names
    def __len__(self):
        return len(self.images)


class Attacker:
    cifar_100_mean = [0.5070, 0.4865, 0.4409]
    cifar_100_std = [0.2673, 0.2564, 0.2761]

    def __init__(self, args):
        self.args = args
        logging.basicConfig(
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            level="INFO",  # "DEBUG" "WARNING" "ERROR"
            stream=sys.stdout,
        )

        self.batch_size = args.batch_size
        self.device = torch.device('cuda' if not args.cpu and torch.cuda.is_available() else 'cpu')

        self.mean = torch.tensor(self.cifar_100_mean).to(self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.cifar_100_std).to(self.device).view(1, 3, 1, 1)

        self.epsilon = args.epsilon_pixels/self.std/255.
        self.alpha = 0.8/self.std/255.

        self.adv_set = AdvDataset(
            args.datadir,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.cifar_100_mean, self.cifar_100_std)
            ]),
        )
        
        self.adv_names = self.adv_set.__getname__()
        self.adv_loader = DataLoader(self.adv_set, batch_size=self.batch_size, shuffle=False)

        logger.info(f'number of images = {self.adv_set.__len__()}')

        self.loss_fn = nn.CrossEntropyLoss()

    def epoch_benign(self, model):
        cuclear()
        with torch.no_grad():
            loader = self.adv_loader
            loss_fn = self.loss_fn
            device = self.device

            model.eval()
            train_acc, train_loss = 0.0, 0.0
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                yp = model(x)
                loss = loss_fn(yp, y)
                train_acc += (yp.argmax(dim=1) == y).sum().item()
                train_loss += loss.item() * x.shape[0]
        return train_acc / len(loader.dataset), train_loss / len(loader.dataset)

    def gen_adv_examples(self, model, attack, victim=None):
        victim = victim if victim is not None else model
        device = self.device
        model.eval()
        victim.eval()
        adv_names = []
        n_data = len(self.adv_loader.dataset)
        train_acc, train_loss = 0.0, 0.0
        for i, (x, y) in enumerate(self.adv_loader):
            cuclear()
            x, y = x.to(device), y.to(device)
            x_adv = attack(model, x, y, self.loss_fn) # obtain adversarial examples
            with torch.no_grad():
                error = (x_adv - x).abs()
                if (error > self.epsilon).any():
                    import pdb; pdb.set_trace()
                
                assert (error <= self.epsilon).all(), f"allowed: {self.epsilon.squeeze().tolist()}, got max: {error.max()} avg: {error.mean()}"
                yp = victim(x_adv)
                loss = self.loss_fn(yp, y)
                train_acc += (yp.argmax(dim=1) == y).sum().item()
                train_loss += loss.item() * x.shape[0]
                # store adversarial examples
                adv_ex = ((x_adv) * self.std + self.mean).clamp(0, 1) # to 0-1 scale
                adv_ex = (adv_ex * 255).clamp(0, 255) # 0-255 scale
                adv_ex = adv_ex.detach().cpu().data.numpy().round() # round to remove decimal part
                adv_ex = adv_ex.transpose((0, 2, 3, 1)) # transpose (bs, C, H, W) back to (bs, H, W, C)
                adv_examples = adv_ex if i == 0 else np.r_[adv_examples, adv_ex]
        return adv_examples, train_acc / n_data, train_loss / n_data

    # create directory which stores adversarial examples
    def create_dir(self, data_dir, adv_dir, adv_examples):
        if os.path.exists(adv_dir) is not True:
            _ = shutil.copytree(data_dir, adv_dir)
        for example, name in zip(adv_examples, self.adv_names):
            im = Image.fromarray(example.astype(np.uint8)) # image pixel value should be unsigned int
            im.save(os.path.join(adv_dir, name))

    def build_model(self, name):
        if name.split("_").index("cifar100") == 0:
            model = torch.hub.load(
                "chenyaofo/pytorch-cifar-models",
                name,
                pretrained=True
            )
        else:
            model = ptcv_get_model(name, pretrained=True)
        model.eval()
        model = model.to(self.device)
        return model

    def solve(self,):        
        model = self.build_model(self.args.model)
        victim = None
        if self.args.victim != self.args.model:
            victim = self.build_model(self.args.victim)

        benign_acc, benign_loss = self.epoch_benign(model)
        logger.info(f'source benign_acc = {benign_acc:.5f}, benign_loss = {benign_loss:.5f}')

        if victim is not None:
            benign_acc, benign_loss = self.epoch_benign(victim)
            logger.info(f'victim benign_acc = {benign_acc:.5f}, benign_loss = {benign_loss:.5f}')

        if self.args.attack == "none":
            return

        logger.info("source: {}".format(model.__class__.__name__))
        if victim is not None:
            logger.info("victim: {}".format(victim.__class__.__name__))
        logger.info("attack: {}".format(self.args.attack))

        def ifgsm(model, x, y, loss_fn, epsilon=self.epsilon, alpha=self.alpha, num_iter=self.args.num_iter):
            x_adv = x.detach().clone()
            for i in range(num_iter):
                x_adv.requires_grad = True
                loss = loss_fn(model(x_adv), y)
                loss.backward()
                grad = x_adv.grad.detach()
                x_adv = x_adv + epsilon * grad.sign()
                x_adv = clamp(x_adv, x, epsilon)
            return x_adv

        def fgsm(model, x, y, loss_fn, epsilon=self.epsilon):
            return ifgsm(model, x, y, loss_fn, epsilon=self.epsilon, alpha=self.epsilon, num_iter=1)

        def input_diversity(x, resize_rate=1.25, diversity_prob=0.5):
            mode = 'nearest' # 'bilinear'
            img_size = x.shape[-1]
            img_resize = int(img_size * resize_rate)

            if resize_rate < 1:
                img_size = img_resize
                img_resize = x.shape[-1]

            rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
            rescaled = F.interpolate(x, size=[rnd, rnd], mode=mode) #, align_corners=False)
            h_rem = img_resize - rnd
            w_rem = img_resize - rnd
            pad_top = torch.randint(low=0, high=h_rem.item()+1, size=(1,), dtype=torch.int32)
            pad_bottom = h_rem - pad_top
            pad_left = torch.randint(low=0, high=w_rem.item()+1, size=(1,), dtype=torch.int32)
            pad_right = w_rem - pad_left

            padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)
            padded = F.interpolate(padded, (img_size, img_size), mode=mode)
            return padded if torch.rand(1) < diversity_prob else x


        def linbp(model, x, y, loss_fn, epsilon=self.epsilon, alpha=self.alpha, num_iter=self.args.num_iter, linbp_layer=self.args.linbp_layer, sgm_lambda=self.args.sgm_lambda):
            x_adv = x.detach().clone()
            for i in range(num_iter):
                x_adv.requires_grad = True
                att_out, ori_mask_ls, conv_out_ls, relu_out_ls, conv_input_ls = linbp_forw_resnet50(model, x_adv, True, linbp_layer)
                loss = loss_fn(att_out, y)
                model.zero_grad()
                grad = linbp_backw_resnet50(x_adv, loss, conv_out_ls, ori_mask_ls, relu_out_ls, conv_input_ls, xp=sgm_lambda)
                x_adv = x_adv.data + alpha * grad.sign()
                x_adv = clamp(x_adv, x, epsilon)
            return x_adv

        if self.args.attack == "fgsm":
            atk_fn = fgsm
        elif self.args.attack == "ifgsm":
            atk_fn = ifgsm
            logger.info("num iters: {}".format(self.args.num_iter))
        elif self.args.attack == "linbp":
            atk_fn = linbp
            assert self.args.linbp_layer != None
            logger.info("num iters: {}".format(self.args.num_iter))
            logger.info("linear BP layer start: {}".format(self.args.linbp_layer))
            logger.info("SGM factor: {}".format(self.args.sgm_lambda))
        else:
            raise NotImplementedError(f"{self.args.attack} attack not implemented")


        adv_examples, acc, loss = self.gen_adv_examples(model, atk_fn, victim)
        logger.info(f'attack {self.args.attack}_acc = {acc:.5f}, {self.args.attack}_loss = {loss:.5f}')

        if self.args.savedir is not None:
            logger.info("validating image constraints...")
            # final_adv = []
            for idx, im in enumerate(adv_examples):
                orig = np.array(Image.open(self.adv_set.images[idx]))
                error = np.absolute(im - orig)
                assert not (error > self.args.epsilon_pixels).any()
                # if (error > self.args.epsilon_pixels).any():
                #     logger.warning(f"allowed: {self.args.epsilon_pixels}, got max: {error.max()} avg: {error.mean()}")
                # final_adv.append(
                #     np.clip(im, orig-args.epsilon_pixels, orig+args.epsilon_pixels))

            self.create_dir(self.args.datadir, self.args.savedir, adv_examples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--datadir", default="./cifar-100_eval")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epsilon-pixels", type=float, default=8)
    # training
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    # checkpoint
    parser.add_argument("--model", default="cifar100_resnet56")
    parser.add_argument("--victim", default='densenet100_k24_cifar100')
    parser.add_argument("--attack", default='none', choices=[
        "none", "fgsm", "ifgsm", "linbp",
    ])
    parser.add_argument("--iters", type=int, dest="num_iter", default=1)
    parser.add_argument("--linbp-layer", default=None) #"3_4")
    parser.add_argument("--sgm-lambda",  type=float, default=0.5)
    parser.add_argument("--savedir", default=None) #'ti-sgm-linbp'

    args = parser.parse_args()

    Attacker(args).solve()
