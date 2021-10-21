import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# general
import shutil
import numpy as np
from qqdm import qqdm as tqdm
import sys
import logging
import argparse
from torchinfo import summary
from pathlib import Path
import os
import glob
from PIL import Image
from pytorchcv.model_provider import get_model as ptcv_get_model

# local
import models
import utils
logger = logging.getLogger(__name__)

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
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

        self.mean = torch.tensor(self.cifar_10_mean).to(device).view(3, 1, 1)
        self.std = torch.tensor(self.cifar_10_std).to(device).view(3, 1, 1)

        self.epsilon = 8/255/std
        self.alpha = 1.0/255/std

        self.adv_set = AdvDataset(
            args.datadir,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.cifar_100_mean, self.cifar_100_std)
            ]),
        )
        
        self.adv_names = adv_set.__getname__()
        self.adv_loader = DataLoader(adv_set, batch_size=self.batch_size, shuffle=False)

        logger.info(f'number of images = {self.adv_set.__len__()}')

        self.loss_fn = nn.CrossEntropyLoss()

    def epoch_benign(self, model):
        loader = self.adv_loader
        loss_fn = self.loss_fn

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
        model.eval()
        victim.eval()
        adv_names = []
        n_data = len(self.adv_loader.dataset)
        train_acc, train_loss = 0.0, 0.0
        for i, (x, y) in enumerate(self.adv_loader):
            x, y = x.to(device), y.to(device)
            x_adv = attack(model, x, y, self.loss_fn) # obtain adversarial examples
            yp = victim(x_adv)
            loss = self.loss_fn(yp, y)
            train_acc += (yp.argmax(dim=1) == y).sum().item()
            train_loss += loss.item() * x.shape[0]
            # store adversarial examples
            adv_ex = ((x_adv) * std + mean).clamp(0, 1) # to 0-1 scale
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

    def try_load_checkpoint(self, model, name=None):
        name = name if name else "checkpoints/checkpoint_last.pt"
        checkpath = Path(name) / name
        if checkpath.exists():
            check = torch.load(checkpath)
            model.load_state_dict(check["model"])
            stats = check["stats"]
            logger.info(
                f"loaded checkpoint {checkpath}: epoch={stats['epoch']} loss={stats['loss']} acc={stats['acc']}")
        else:
            logger.info(f"no checkpoints found at {checkpath}!")
        return model

    def solve(self, attack):
        device = self.device
        model = models.resnet.cifar100_resnet56().to(device)
        model = self.try_load_checkpoint(model, name=self.args.resume)
        victim = ptcv_get_model('resnext29_16x64d_cifar100', pretrained=True).to(device)
        model.eval()
        victim.eval()

        benign_acc, benign_loss = self.epoch_benign(model)
        logger.info(f'benign_acc = {benign_acc:.5f}, benign_loss = {benign_loss:.5f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--datadir", default="./cifar-100_eval")
    parser.add_argument("--num-workers", type=int, default=2)
    # training
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--max-epoch", type=int, default=90)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--cuda", action="store_true")
    # checkpoint
    parser.add_argument("--resume", default="")

    args = parser.parse_args()

    def fgsm(model, x, y, loss_fn, epsilon=epsilon):
        x_adv = x.detach().clone()
        x_adv.requires_grad = True
        loss = loss_fn(model(x_adv), y)
        loss.backward()
        grad = x_adv.grad.detach()
        x_adv = x_adv + epsilon * grad.sign()
        return x_adv

    def ifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20):
        x_adv = x.detach().clone()
        for i in range(num_iter):
            x_adv = fgsm(model, x_adv, y, loss_fn, epsilon=alpha)
            delta = torch.clip(x_adv - x, min=-epsilon, max=epsilon)
            x_adv = (x + delta).detach()
        return x_adv

    Attacker(args).solve(fgsm)
