# Standard Library
from tqdm import tqdm

# PyTorch
import torch

# utils
from utils import visualize
from utils.metrics import Summary, AverageMeter

# trainers
from trainers.train_base import TrainBase


class TrainSatMAE(TrainBase):
    def get_loss(self, images, labels):
        images = images[:, :, 16:-16, 16:-16]
        labels = labels[:, :, 16:-16, 16:-16]
        outputs = self.model(images)
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        images = images[:, :, 16:-16, 16:-16]
        labels = labels[:, :, 16:-16, 16:-16]
        visualize.visualize(
            x=images,
            y=labels,
            y_pred=outputs.detach().cpu().numpy(),
            images=5,
            channel_first=True,
            vmin=0,
            save_path=f"{self.out_folder}/{name}.png"
        )

    def v_loop(self, epoch):

        # set model to evaluation mode
        self.model.eval()

        # Initialize the validation loss meter
        val_loss_meter = AverageMeter('val_loss', ':.4f', Summary.AVERAGE)

        # Initialize the progress bar for training
        if self.RANK == 0:
            val_pbar = tqdm(total=len(self.val_loader), desc=f"Epoch {epoch + 1}/{self.epochs}")

        with torch.no_grad():
            for j, (images, labels) in enumerate(self.val_loader):
                # Move inputs and targets to the device (GPU)
                images, labels = images.to(self.device), labels.to(self.device)

                # get loss
                loss = self.get_loss(images, labels)

                # update local loss meter
                val_loss_meter.update(loss.item(), 1)

                # display progress on console
                if self.RANK == 0:
                    val_pbar.update(1)

            # Close the progress bar
            if self.RANK == 0:
                val_pbar.close()

            if self.visualise_validation and self.RANK == 0:
                outputs = self.model(images[:, :, 16:-16, 16:-16])

                if type(outputs) is tuple:
                    outputs = outputs[0]

                self.val_visualize(
                    images.detach().cpu().numpy(),
                    labels.detach().cpu().numpy(),
                    outputs.detach().cpu().numpy(),
                    name=f'/val_images/val_{epoch}'
                )

        # synchronize the epoch's losses across devices
        val_loss_meter.all_reduce()

        return val_loss_meter.avg