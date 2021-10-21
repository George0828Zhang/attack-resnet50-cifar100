import torch
import torch.nn as nn
from torchvision import datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# general
import shutil
import numpy as np
# from qqdm import qqdm as tqdm
from tqdm import tqdm
import sys
import logging
import argparse
from torchinfo import summary
import wandb
from pathlib import Path

# local
import models
import utils
logger = logging.getLogger(__name__)


class Trainer:
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
        if args.use_wandb:
            wandb.init(project="resnet50-cifar100", name=Path(args.savedir).stem, config=args)

        # self.model = models.resnet.resnet50(num_classes=100)
        self.model = models.resnet.resnet56()
        summary(self.model)

        self.train_set = datasets.CIFAR100(
            args.datadir,
            train=True,
            transform=transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.cifar_100_mean, self.cifar_100_std)
            ]),
            download=True
        )
        self.val_set = datasets.CIFAR100(
            args.datadir,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.cifar_100_mean, self.cifar_100_std)
            ]),
            download=True
        )
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        self.device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

        self.criterion = utils.LabelSmoothedCrossEntropyCriterion(
            smoothing=args.label_smoothing,
        )

        self.model.to(self.device)
        self.criterion.to(self.device)
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=0.256,
            momentum=0.875,
            weight_decay=3.0517578125e-05,
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     self.optimizer,
        #     T_0=10,
        #     T_mult=2,
        #     eta_min=0,
        # )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
        )
        self.initialize_stats()

    def initialize_stats(self,):
        self.epoch = 0

    def train_one_epoch(self,):
        # gradient accumulation: update every accum_steps samples
        itr = utils.GroupedIterator(self.train_loader, self.args.accum_steps)

        stats = {"loss": []}
        scaler = GradScaler()  # automatic mixed precision (amp)

        model = self.model
        criterion = self.criterion
        optimizer = self.optimizer
        scheduler = self.scheduler

        model.train()
        n_iters = len(itr)
        self.epoch += 1
        progress = tqdm(itr, desc=f"train epoch {self.epoch}", leave=False)
        for i, samples in enumerate(progress):
            model.zero_grad()
            accum_loss = 0.
            sample_size = 0.
            # gradient accumulation: update every accum_steps samples
            for j, sample in enumerate(samples):
                if j == 1:
                    # emptying the CUDA cache after the first step can reduce the chance of OOM
                    torch.cuda.empty_cache()

                sample = utils.move_to_cuda(sample, device=self.device)
                inputs, target = sample
                sample_size += inputs.size(0)

                # mixed precision training
                with autocast():
                    net_output = model.forward(inputs)
                    lprobs = net_output.log_softmax(-1)
                    loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1))

                    # logging
                    accum_loss += loss.item()
                    # back-prop
                    scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            utils.multiply_grads(optimizer, 1 / (sample_size or 1.0))
            gnorm = nn.utils.clip_grad_norm_(model.parameters(), self.args.clip_norm)

            scaler.step(optimizer)
            scaler.update()
            #scheduler.step(self.epoch + i / n_iters)

            # logging
            loss_print = accum_loss / sample_size
            stats["loss"].append(loss_print)
            infos = {
                "train/loss": loss_print,
                "train/grad_norm": gnorm.item(),
                # "train/lr": scheduler.get_last_lr()[0],
                "train/sample_size": sample_size,
            }
            progress.set_postfix(**utils.floatdict2str(infos))
            if self.args.use_wandb:
                wandb.log(infos)

        loss_print = np.mean(stats["loss"])
        logger.info(f"training loss: {loss_print:.4f}")
        return stats

    def validate(self, log_to_wandb=True):
        logger.info('begin validation')
        itr = self.val_loader

        stats = {"loss": [], "acc": []}

        model = self.model
        criterion = self.criterion

        model.eval()
        progress = tqdm(itr, desc="validation", leave=False)
        with torch.no_grad():
            for i, sample in enumerate(progress):
                # validation loss
                sample = utils.move_to_cuda(sample, device=self.device)
                inputs, target = sample
                sample_size = inputs.size(0)
                net_output = model.forward(inputs)
                lprobs = net_output.log_softmax(-1)
                loss = criterion(lprobs.view(-1, lprobs.size(-1)), target.view(-1)) / sample_size
                acc = (lprobs.argmax(dim=-1) == target).cpu().tolist()
                stats["loss"].append(loss.item())
                stats["acc"].extend(acc)
                progress.set_postfix(
                    **utils.floatdict2str({
                        "valid/loss": loss.item(),
                        "valid/acc": np.mean(acc),
                    })
                )

        stats["loss"] = np.mean(stats["loss"])
        stats["acc"] = np.mean(stats["acc"])

        if self.args.use_wandb and log_to_wandb:
            wandb.log({
                "valid/loss": stats["loss"],
                "valid/acc": stats["acc"],
            }, commit=False)

        # show results
        logger.info(f"validation loss:\t{stats['loss']:.4f}")
        logger.info(f"validation acc:\t{stats['acc']:.4f}")
        return stats

    def validate_and_save(self, save=True):
        stats = self.validate(log_to_wandb=True)
        acc = stats['acc']
        loss = stats['loss']
        if save:
            # save epoch checkpoints
            savedir = Path(self.args.savedir).absolute()
            savedir.mkdir(parents=True, exist_ok=True)

            epoch = self.epoch
            model = self.model
            optimizer = self.optimizer
            scheduler = self.scheduler

            check = {
                "model": model.state_dict(),
                "stats": {"epoch": epoch, "acc": acc, "loss": loss},
                "optim": optimizer.state_dict(),
                "sched": scheduler.state_dict(),
            }
            torch.save(check, savedir / f"checkpoint{epoch}.pt")
            shutil.copy(savedir / f"checkpoint{epoch}.pt", savedir / "checkpoint_last.pt")
            logger.info(f"saved epoch checkpoint: {savedir}/checkpoint{epoch}.pt")

            # get best valid acc
            if getattr(self, "best_bleu", 0) < acc:
                self.best_bleu = acc
                torch.save(check, savedir / "checkpoint_best.pt")

            del_file = savedir / f"checkpoint{epoch - self.args.keep_last_epochs}.pt"
            if del_file.exists():
                del_file.unlink()

        return stats

    def try_load_checkpoint(self, name=None, load_optim=True, load_sched=True):
        name = name if name else "checkpoint_last.pt"
        checkpath = Path(self.args.savedir) / name
        if checkpath.exists():
            check = torch.load(checkpath)
            self.model.load_state_dict(check["model"])
            stats = check["stats"]
            if load_optim:
                self.optimizer.load_state_dict(check["optim"])
            if load_sched:
                self.scheduler.load_state_dict(check["sched"])
            self.epoch = stats['epoch']
            logger.info(
                f"loaded checkpoint {checkpath}: epoch={stats['epoch']} loss={stats['loss']} acc={stats['acc']}")
        else:
            logger.info(f"no checkpoints found at {checkpath}!")

    def solve(self):
        logger.info("model: {}".format(self.model.__class__.__name__))
        logger.info("criterion: {}".format(self.criterion.__class__.__name__))
        logger.info("optimizer: {}".format(self.optimizer.__class__.__name__))
        logger.info("scheduler: {}".format(self.scheduler.__class__.__name__))
        logger.info(
            "num. model params: {:,} (num. trained: {:,})".format(
                sum(p.numel() for p in self.model.parameters()),
                sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            )
        )
        logger.info(f"batch size = {self.args.batch_size}, accumulate steps = {self.args.accum_steps}")
        self.try_load_checkpoint(name=self.args.resume)
        while self.epoch <= self.args.max_epoch:
            # train for one epoch
            self.train_one_epoch()
            self.scheduler.step(self.validate_and_save()["loss"])
            logger.info("end of epoch {}".format(self.epoch))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--datadir", default="./cifar-100_train")
    parser.add_argument("--num-workers", type=int, default=2)
    # logging
    parser.add_argument("--use-wandb", action="store_true")
    # training
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--clip-norm", type=float, default=10.0)
    parser.add_argument("--max-epoch", type=int, default=90)
    parser.add_argument("--start-epoch", type=int, default=1)
    parser.add_argument("--cuda", action="store_true")
    # checkpoint
    parser.add_argument("--keep-last-epochs", type=int, default=1)
    parser.add_argument("--resume", default="")
    parser.add_argument("--savedir", default="./checkpoints")

    # optim
    parser.add_argument("--label-smoothing", type=float, default=0.1)

    args = parser.parse_args()

    Trainer(args).solve()
