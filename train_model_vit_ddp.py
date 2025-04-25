import argparse
import os
import random
import sys
from datetime import datetime
from enum import Enum

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.majortom import MajorTOM
from model.phileo_vit import PhilEO_ViT
from loss.losses import foundation_loss
from utils import cosine_scheduler


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """ Computes and stores the average and current value """

    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'

        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)


class PytorchDistributedTrainer:
    def __init__(self):
        self.init_dist()
        self.init_args()
        self.init_seed()
        self.init_model()
        self.init_loader()
        self.init_optim()
        self.init_loss()
        self.init_wandb()

    def init_args(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--random_seed', type=int, default=14)
        parser.add_argument('--num_epochs', type=int, default=100000)
        parser.add_argument('--batch_size', type=int, default=16)
        parser.add_argument('--learning_rate', type=float, default=3e-5)
        parser.add_argument('--learning_rate_end', type=float, default=1e-8)
        parser.add_argument('--img_size', type=int, default=128)
        parser.add_argument('--base_path', type=str, default='/path/to/major-tom')
        parser.add_argument('--model_name', type=str, default='phileo_vit')
        parser.add_argument('--save_models', type=bool, default=True)
        parser.add_argument('--warmup_epochs', type=int, default=0)
        parser.add_argument('--warmup_lr_start', type=float, default=1e-6)
        parser.add_argument('--weight_decay', type=float, default=1e-2)
        parser.add_argument('--es_patience', type=int, default=30)
        parser.add_argument('--es_delta', type=float, default=0.)
        self.params = parser.parse_args()
        self.params.batch_size = self.params.batch_size
        self.params.learning_rate = self.params.learning_rate * np.sqrt(self.WORLD_SIZE)
        self.params.learning_rate_end = self.params.learning_rate_end * np.sqrt(self.WORLD_SIZE)

    def init_seed(self):
        np.random.seed(self.params.random_seed)
        os.environ['PYTHONHASHSEED'] = str(self.params.random_seed)
        random.seed(self.params.random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(self.params.random_seed)
        torch.manual_seed(self.params.random_seed)

    def init_dist(self):
        if 'LOCAL_RANK' in os.environ:
            self.WORLD_SIZE = int(os.environ['WORLD_SIZE'])
            self.RANK = int(os.environ['RANK'])
            self.LOCAL_RANK = int(os.environ['LOCAL_RANK'])
        elif 'OMPI_COMM_WORLD_LOCAL_RANK' in os.environ:
            self.WORLD_SIZE = int(os.environ['OMPI_COMM_WORLD_SIZE'])
            self.RANK = int(os.environ['OMPI_COMM_WORLD_RANK'])
            self.LOCAL_RANK = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        else:
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "23456"
            self.WORLD_SIZE = 1
            self.RANK = 0
            self.LOCAL_RANK = 0
            self.log("Can't find the environment variables!")

        dist.init_process_group(backend='nccl', rank=self.RANK, world_size=self.WORLD_SIZE)
        self.DEVICE = 'cuda:{}'.format(self.LOCAL_RANK)
        torch.cuda.set_device(self.LOCAL_RANK)

    def init_model(self):
        model = PhilEO_ViT(
            input_dim=10,
            chw=(10, 128, 128),
            patch_size=4,
            embed_dim=512,
            depth=32,
            num_heads=16,
            mlp_ratio=4,
            norm_layer=nn.LayerNorm,
            latent_dim=1024,
            dropout=None,
            activation=nn.LeakyReLU()
        )

        # Convert batchnorm to syncbatchnorm
        # https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # , process_group)
        model = self.init_checkpoint(model)
        self.model = DDP(model.to(self.DEVICE), device_ids=[self.LOCAL_RANK], output_device=self.LOCAL_RANK)

    def init_loader(self):
        self.augment = transforms.Compose(
            [transforms.RandomVerticalFlip(p=0.5), transforms.RandomHorizontalFlip(p=0.5)]
        )
        self.augment_drops = transforms.Compose(
            [transforms.RandomErasing(p=1.0, scale=(0.25, 0.50), ratio=(0.3, 3.3), value="random", inplace=False)]
        )

        train_set = MajorTOM(
            MajorTOM_dataset_path=f"{self.params.base_path}/MajorTOM/",
            MajorTOM_pickle_path=f"{self.params.base_path}/MajorTOM/MajorTOM_Paths.pkl",
            label_folder_path=f"{self.params.base_path}/data_static/",
            patch_size=self.params.img_size,
            transform=self.augment,
        )

        train_sampler = DistributedSampler(
            dataset=train_set, rank=self.RANK, num_replicas=self.WORLD_SIZE, shuffle=True,
            drop_last=True
        )
        self.train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.params.batch_size,
            sampler=train_sampler,
            pin_memory=True,
        )

    def init_optim(self):
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.params.learning_rate,
            weight_decay=self.params.weight_decay,
        )
        self.scaler = torch.amp.GradScaler()

        self.lr_schedule_values = cosine_scheduler(
            self.params.learning_rate,
            self.params.learning_rate_end,
            self.params.num_epochs + self.params.warmup_epochs,
            self.params.warmup_epochs,
            self.params.warmup_lr_start,
        )

    def init_loss(self):
        self.patience = self.params.es_patience
        self.counter = 0
        self.delta = self.params.es_delta

    def init_checkpoint(self, model):
        checkpoint_path = os.path.join(self.params.base_path, 'checkpoints')

        files = []
        for filename in os.listdir(checkpoint_path):
            if self.params.model_name in filename:
                files.append(filename)

        if files:
            files.sort(reverse=True)
            latest_checkpoint = os.path.join(checkpoint_path, files[0])
            self.log(f"Loading checkpoint from {latest_checkpoint}")
            checkpoint = torch.load(latest_checkpoint, weights_only=True)
            model.load_state_dict(checkpoint["model"])
            self.epoch = checkpoint["epoch"]
            self.best_score = checkpoint["best_score"]
            self.loss_min = checkpoint["loss_min"]
        else:
            self.log("Checkpoint not found, training from scratch")
            self.epoch = 0
            self.best_score = None
            self.loss_min = np.inf

        return model

    def init_wandb(self):
        if self.RANK == 0:
            os.environ["WANDB_SILENT"] = "true"
            wandb.require("service")
            date_str = datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
            self.wandb = wandb.init(
                mode="offline", # keep this line if you want to use wandb offline, can always sync later!
                project="PROJECT_NAME",
                name=f"{self.params.model_name} - {date_str}",
                config=vars(self.params),
                settings=wandb.Settings(init_timeout=120),
            )
            self.wandb.watch(self.model, log="all")


    def early_stopper(self, loss, epoch):
        score = -loss
        if self.best_score is None:
            self.best_score = score
            self.save_model(loss, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.log(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.cleanup()
                self.log('Finished Training due to Early Stopping')
                sys.exit(0)
        else:
            self.best_score = score
            self.save_model(loss, epoch)
            self.counter = 0

    def save_model(self, loss, epoch):
        """ Saves model when validation loss decrease. """

        path = f"{self.params.base_path}/checkpoints/{self.params.model_name}_epoch_{epoch}.pt"

        self.log(f'Validation loss decreased ({self.loss_min:.4f} --> {loss:.4f}). Saving model ...')
        torch.save(
            {
                "model": self.model.module.state_dict(),
                "epoch": epoch,
                "best_score": self.best_score,
                "loss_min": self.loss_min,
            },
            path,
        )
        self.loss_min = loss

    def cleanup(self):
        dist.barrier()
        dist.destroy_process_group()
        if self.RANK == 0:
            self.wandb.finish()

    def log(self, s, nl=True):
        if int(self.RANK) == 0:
            print(s, end='\n' if nl else '')

    def fit(self):
        for epoch in range(self.epoch, self.params.num_epochs + self.params.warmup_epochs):
            self.log('Epoch: {}, Training ...'.format(epoch))
            self.train_loader.sampler.set_epoch(epoch)
            for param_group in self.optimizer.param_groups:
                param_group['flr'] = self.lr_schedule_values[epoch]

            loss_accum = {
                "loss": AverageMeter('Loss', ':.4f', Summary.AVERAGE),
                "rec": AverageMeter('Rec', ':.4f', Summary.AVERAGE),
                "xy": AverageMeter('XY', ':.4f', Summary.AVERAGE),
                "cl": AverageMeter('CL', ':.4f', Summary.AVERAGE),
                "b": AverageMeter('B', ':.4f', Summary.AVERAGE),
                "lc": AverageMeter('LC', ':.4f', Summary.AVERAGE),
                "sim": AverageMeter('SIM', ':.4f', Summary.AVERAGE),
            }
            to_log = {
                "loss": 0.0,
                "rec": 0.0,
                "xy": 0.0,
                "cl": 0.0,
                "b": 0.0,
                "lc": 0.0,
                "sim": 0.0,
            }

            self.model.train()

            for i, data in enumerate(self.train_loader):
                inputs, labels = data

                inputs = torch.cat([inputs, torch.ones_like(inputs[:, :1, :, :])], dim=1)
                inputs = inputs.to(self.DEVICE)
                for label in labels.keys():
                    labels[label] = labels[label].to(self.DEVICE)

                inputs_augs_1 = self.augment(inputs)
                inputs_aug_1, inputs_aug_1_mask = inputs_augs_1[:, :-1, :, :], inputs_augs_1[:, -1:, :, :]

                inputs_augs_2 = self.augment(inputs)
                inputs_aug_2, inputs_aug_2_mask = inputs_augs_2[:, :-1, :, :], inputs_augs_2[:, -1:, :, :]

                inputs_drop1 = self.augment_drops(inputs_aug_1)
                inputs_drop2 = self.augment_drops(inputs_aug_2)

                self.optimizer.zero_grad()
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    og_recon, og_emb, _og_emb_cnn, _og_decode, og_preds = self.model(inputs_drop1)
                    aug_recon, aug_emb, _aug_emb_cnn, _aug_decode, aug_preds = self.model(inputs_drop2)

                    loss, log = foundation_loss(
                        og_recon * inputs_aug_1_mask,
                        og_emb,
                        og_preds,
                        aug_recon * inputs_aug_2_mask,
                        aug_emb,
                        aug_preds,
                        inputs_aug_1,
                        inputs_aug_2,
                        labels,
                    )

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                for key in loss_accum.keys():
                    loss_accum[key].update(log[key], inputs.size(0))

            for key in to_log.keys():
                to_log[key] = loss_accum[key].all_reduce()

            if self.RANK == 0:
                for name, val in to_log.items():
                    self.wandb.log({name: val, "epoch": epoch})

            if self.RANK == 0:
                self.early_stopper(to_log["loss"], epoch)

        self.log('Finished Training')

        self.cleanup()


if __name__ == "__main__":
    trainer = PytorchDistributedTrainer()
    trainer.fit()
