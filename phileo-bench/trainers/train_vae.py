# Standard Library
import os
from tqdm import tqdm


# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms

# utils
from utils import visualize
from utils.metrics import Summary, AverageMeter

# trainers
from trainers.train_base import TrainBase


class TrainVAE(TrainBase):

    def __init__(self, *args, **kwargs):  # 2048 512
        super(TrainVAE, self).__init__(*args, **kwargs)
        self.CE_loss = nn.CrossEntropyLoss()
        self.MSE_loss = nn.MSELoss()
        self.augmentations = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), value='random'),
            transforms.RandomApply([
                transforms.RandomResizedCrop(128, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=2, antialias=True),
                transforms.RandomRotation(degrees=20),
                transforms.GaussianBlur(kernel_size=3),
            ], p=0.2),
            # transforms.ColorJitter(
            #     brightness=0.25,
            #     contrast=0.25,
            #     saturation=0.5,
            #     hue=0.05,),
            # transforms.RandomAdjustSharpness(0.5, p=0.2),
            # transforms.RandomAdjustSharpness(1.5, p=0.2),
         ])

    def reconstruction_loss(self, reconstruction, original):
        # Binary Cross-Entropy with Logits Loss
        batch_size = original.size(0)

        # BCE = F.binary_cross_entropy_with_logits(
        #   reconstruction.reshape(batch_size, -1),
        #   original.reshape(batch_size, -1),
        #   reduction='mean',
        #  )

        MSE = F.mse_loss(
            reconstruction.reshape(batch_size, -1),
            original.reshape(batch_size, -1),
            reduction='mean',
        )
        # KLDIV = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        return MSE

    def similarity_loss(self, embeddings, embeddings_aug):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        embeddings_aug = F.normalize(embeddings_aug, p=2, dim=1)
        loss_cos = 1 - F.cosine_similarity(embeddings, embeddings_aug).mean()

        return loss_cos

    def cr_loss(self, mu, logvar, mu_aug, logvar_aug, gamma=1e-3, eps=1e-6):
        std_orig = logvar.exp() + eps
        std_aug = logvar_aug.exp() + eps

        _cr_loss = 0.5 * torch.sum(
            2 * torch.log(std_orig / std_aug) - 1 + (std_aug ** 2 + (mu_aug - mu) ** 2) / std_orig ** 2, dim=1).mean()
        cr_loss = _cr_loss * gamma

        return cr_loss

    def get_loss_aug(self, images, aug_images, labels):

        reconstruction, meta_data, latent = self.model(images)
        reconstruction_aug, meta_data_aug, latent_aug = self.model(aug_images)

        reconstruction_loss = (self.reconstruction_loss(reconstruction=reconstruction, original=images) +
                               self.reconstruction_loss(reconstruction=reconstruction_aug, original=aug_images)) / 2

        kg_labels = labels[:, :31]
        coord_labels = labels[:, 31:34]
        time_labels = labels[:, 34:]
        coord_out, time_out, kg_out = meta_data
        coord_out_aug, time_out_aug, kg_out_aug = meta_data_aug

        kg_loss = (self.CE_loss(kg_out, kg_labels) + self.CE_loss(kg_out_aug, kg_labels)) / 2
        coord_loss = (self.MSE_loss(coord_out, coord_labels) + self.MSE_loss(coord_out_aug, coord_labels)) / 2
        time_loss = (self.MSE_loss(time_out, time_labels) + self.MSE_loss(time_out_aug, time_labels)) / 2

        contrastive_loss = self.similarity_loss(latent, latent_aug)

        loss = reconstruction_loss + kg_loss + coord_loss + time_loss + contrastive_loss
        outputs = (reconstruction, meta_data, latent)

        return loss, reconstruction_loss, kg_loss, coord_loss, time_loss, contrastive_loss, outputs

    def get_loss(self, images, labels):
        reconstruction, meta_data, scale_skip_loss = self.model(images)

        reconstruction_loss = self.reconstruction_loss(reconstruction=reconstruction, original=images)

        kg_labels = labels[:, :31]
        coord_labels = labels[:, 31:34]
        time_labels = labels[:, 34:]
        coord_out, time_out, kg_out = meta_data

        kg_loss = self.CE_loss(kg_out, kg_labels)
        coord_loss = self.MSE_loss(coord_out, coord_labels)
        time_loss = self.MSE_loss(time_out, time_labels)

        # loss = 0.5*reconstruction_loss + 0.25*kg_loss + 0.125*coord_loss + 0.125*time_loss + scale_skip_loss
        loss = reconstruction_loss + kg_loss + coord_loss + time_loss + scale_skip_loss
        outputs = (reconstruction, meta_data, scale_skip_loss)

        return loss, reconstruction_loss, kg_loss, coord_loss, time_loss, scale_skip_loss, outputs

    def t_loop(self, epoch, s):

        # set model to train mode
        self.model.train()

        # Initialize the training loss meter
        train_loss_meter = AverageMeter('train_loss', ':.4f', Summary.AVERAGE)
        # train_reconstruction_loss_meter = AverageMeter('train_rec_loss', ':.4f', Summary.AVERAGE)
        # train_kg_loss_meter = AverageMeter('train_kg_loss', ':.4f', Summary.AVERAGE)
        # train_coord_loss_meter = AverageMeter('train_coord_loss', ':.4f', Summary.AVERAGE)
        # train_time_loss_meter = AverageMeter('train_time_loss', ':.4f', Summary.AVERAGE)
        # train_scale_skip_loss_meter = AverageMeter('train_scale_skip_loss', ':.4f', Summary.AVERAGE)

        # Initialize the progress bar for training
        if self.RANK == 0:
            train_pbar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.epochs}")

        # loop training through batches
        for i, (images, labels) in enumerate(self.train_loader):
            # Move inputs and targets to the device (GPU)
            images, labels = images.to(self.device), labels.to(self.device)

            # Zero the gradients
            self.optimizer.zero_grad()

            # get loss
            with autocast(dtype=torch.float16):
                # (
                #    loss,
                #    reconstruction_loss,
                #    kg_loss,
                #    coord_loss,
                #    time_loss,
                #    scale_skip_loss,
                #    outputs
                # ) = self.get_loss(images, labels)
                loss, _, _, _, _, _, outputs = self.get_loss(images, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            train_loss_meter.update(loss.item(), 1)
            # train_reconstruction_loss_meter.update(reconstruction_loss.item(), 1)
            # train_kg_loss_meter.update(kg_loss.item(), 1)
            # train_coord_loss_meter.update(coord_loss.item(), 1)
            # train_time_loss_meter.update(time_loss.item(), 1)
            # train_scale_skip_loss_meter.update(scale_skip_loss.item(), 1)

            # # Update the scheduler
            if self.lr_scheduler == 'cosine_annealing':
                s.step()

            # visualize training
            if (i % 10000) == 0 and i != 0 and self.RANK == 0:
                self.val_visualize(images, labels, outputs, name=f'/val_images/train_{epoch}_{i}')
                model_sd = self.model.state_dict()
                torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_ckpt.pt"))

            # display progress on console
            if self.RANK == 0:
                train_pbar.update(1)

        # Close the progress bar
        if self.RANK == 0:
            train_pbar.close()

        # synchronize the epoch's losses across devices
        train_loss_meter.all_reduce()
        # train_reconstruction_loss_meter.all_reduce()
        # train_kg_loss_meter.all_reduce()
        # train_coord_loss_meter.all_reduce()
        # train_time_loss_meter.all_reduce()
        # train_scale_skip_loss_meter.all_reduce()

        return train_loss_meter.avg

    def v_loop(self, epoch):

        # set model to evaluation mode
        self.model.eval()

        # Initialize the training loss meter
        val_loss_meter = AverageMeter('val_loss', ':.4f', Summary.AVERAGE)
        # val_reconstruction_loss_meter = AverageMeter('val_rec_loss', ':.4f', Summary.AVERAGE)
        # val_kg_loss_meter = AverageMeter('val_kg_loss', ':.4f', Summary.AVERAGE)
        # val_coord_loss_meter = AverageMeter('val_coord_loss', ':.4f', Summary.AVERAGE)
        # val_time_loss_meter = AverageMeter('val_time_loss', ':.4f', Summary.AVERAGE)
        # val_scale_skip_loss_meter = AverageMeter('val_scale_skip_loss', ':.4f', Summary.AVERAGE)

        # Initialize the progress bar for training
        if self.RANK == 0:
            val_pbar = tqdm(total=len(self.val_loader), desc=f"Epoch {epoch + 1}/{self.epochs}")

        with torch.no_grad():
            for j, (images, labels) in enumerate(self.val_loader):
                # Move inputs and targets to the device (GPU)
                images, labels = images.to(self.device), labels.to(self.device)

                # (
                #    loss,
                #    reconstruction_loss,
                #    kg_loss,
                #    coord_loss,
                #    time_loss,
                #    scale_skip_loss,
                #    outputs
                # ) = self.get_loss(images, labels)
                loss, _, _, _, _, _, outputs = self.get_loss(images, labels)

                val_loss_meter.update(loss.item(), 1)
                # val_reconstruction_loss_meter.update(reconstruction_loss.item(), 1)
                # val_kg_loss_meter.update(kg_loss.item(), 1)
                # val_coord_loss_meter.update(coord_loss.item(), 1)
                # val_time_loss_meter.update(time_loss.item(), 1)
                # val_scale_skip_loss_meter.update(scale_skip_loss.item(), 1)

                # display progress on console
                if self.RANK == 0:
                    val_pbar.update(1)

        # Close the progress bar
        if self.RANK == 0:
            val_pbar.close()

        if self.visualise_validation and self.RANK == 0:
            self.val_visualize(images, labels, outputs, name=f'/val_images/val_{epoch}')

        # synchronize the epoch's losses across devices
        val_loss_meter.all_reduce()
        # val_reconstruction_loss_meter.all_reduce()
        # val_kg_loss_meter.all_reduce()
        # val_coord_loss_meter.all_reduce()
        # val_time_loss_meter.all_reduce()
        # val_scale_skip_loss_meter.all_reduce()

        return val_loss_meter.avg

    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize_vae(
            images=images,
            labels=labels,
            outputs=outputs,
            num_images=5,
            channel_first=True,
            save_path=f"{self.out_folder}/{name}.png"
        )
