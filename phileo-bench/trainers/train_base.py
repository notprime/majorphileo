# Standard Library
#import matplotlib
#matplotlib.use('TKAgg')
import os
from tqdm import tqdm
from matplotlib import pyplot as plt


# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau, MultiStepLR
import json

# utils
from utils import visualize
from utils.metrics import Summary, AverageMeter


class TrainBase:

    def __init__(
            self,
            RANK,
            wandb,
            model: nn.Module,
            device: torch.device,
            train_loader: DataLoader,
            val_loader: DataLoader,
            test_loader: DataLoader,
            epochs: int = 50,
            es_patience: int = 25,
            es_delta: float = 0.0,
            lr: float = 0.001,
            lr_scheduler: str = None,
            metrics: list = None,
            name: str = "model",
            out_folder: str = "trained_models/",
            visualise_validation: bool = True,
            warmup_epochs: int = 5,
            warmup_gamma: int = 10
    ):

        self.RANK = RANK
        self.wandb = wandb

        self.test_loss = None
        self.last_epoch = None
        self.best_sd = None
        self.epochs = epochs
        self.es_patience = es_patience
        self.es_delta = es_delta
        self.learning_rate = lr
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.metrics = metrics
        self.lr_scheduler = lr_scheduler
        self.warmup_epochs = warmup_epochs
        self.name = name
        self.out_folder = out_folder
        self.visualise_validation = visualise_validation

        if self.RANK == 0:
            os.makedirs(self.out_folder, exist_ok=True)
            if visualise_validation:
                os.makedirs(f'{self.out_folder}/val_images', exist_ok=True)

        self.scaler, self.optimizer = self.set_optimizer()
        self.criterion = self.set_criterion()
        self.scheduler = self.set_scheduler()

        if self.warmup_epochs > 0:
            multistep_milestone = list(range(1, self.warmup_epochs + 1))
            self.scheduler_warmup = MultiStepLR(self.optimizer, milestones=multistep_milestone, gamma=warmup_gamma)

        # initialize torch device
        # torch.set_default_device(self.device)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        else:
            print(F"Rank: {self.RANK}: No CUDA device available.")

        # init useful variables
        self.best_epoch = 0
        self.best_loss = None
        self.best_model_state = model.state_dict().copy()
        self.epochs_no_improve = 0

        # used for plots
        self.tl = []
        self.vl = []
        self.e = []
        self.lr = []

    def is_rank_zero(self):
        return self.RANK == 0

    def print_distributed(self, s, nl=True):
        if self.RANK == 0:
            print(s, end='\n' if nl else '')

    def set_optimizer(self):
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, eps=1e-06)
        #scaler = GradScaler()
        scaler = torch.amp.GradScaler('cuda')

        # Save the initial learning rate in optimizer's param_groups
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = self.learning_rate

        return scaler, optimizer

    def set_criterion(self):
        return nn.MSELoss()

    def set_scheduler(self):
        if self.lr_scheduler == 'cosine_annealing':
            scheduler = CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=20,
                T_mult=2,
                eta_min=0.000001,
                last_epoch=self.epochs - 1,
            )
        elif self.lr_scheduler == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.1,
                patience=6,
                min_lr=1e-6
            )
        else:
            scheduler = None

        return scheduler

    def get_loss(self, images, labels):
        outputs = self.model(images)
        #outputs = torch.sigmoid(outputs)
        #print("--- DEBUGGING: APPLYING SIGMOID FOR ROAD REGRESSION IN get_loss method ---")
        loss = self.criterion(outputs, labels)
        return loss

    def get_metrics_meters(self):
        metrics_meters = {
            "mse": AverageMeter('mse', ':.4f', Summary.AVERAGE),
            "baseline_mse": AverageMeter('baseline_mse', ':.4f', Summary.AVERAGE),
            "mae": AverageMeter('mae', ':.4f', Summary.AVERAGE),
            "mave": AverageMeter('mave', ':.4f', Summary.AVERAGE),

            "tp": AverageMeter('tp', ':.4f', Summary.AVERAGE),
            "fp": AverageMeter('fp', ':.4f', Summary.AVERAGE),
            "fn": AverageMeter('fn', ':.4f', Summary.AVERAGE),
            "tn": AverageMeter('tn', ':.4f', Summary.AVERAGE),
        }
        return metrics_meters

    def update_metrics_meters(self, metrics_meters, labels, images):

        outputs = self.model(images)

        # regression metrics
        error = outputs - labels
        squared_error = error ** 2
        mse = squared_error.mean().item()   # batch MSE
        mae = error.abs().mean().item()     # batch MAE
        mave = torch.mean(torch.abs(outputs.mean(dim=(1, 2)) - labels.mean(dim=(1, 2)))).item()     # BATCH MAVE
        zero_model_mse = (labels ** 2).mean().item()    # zero-model base MSE

        # regression metrics disguised as classification
        threshold = 0.5
        label_classification = (labels > threshold).type(torch.int8)
        output_classification = (outputs > threshold).type(torch.int8)

        num_pixels = label_classification.numel()
        tp = torch.count_nonzero((label_classification == 1) & (output_classification == 1)).item() / num_pixels    # batch TP %
        fp = torch.count_nonzero((label_classification == 1) & (output_classification == 0)).item() / num_pixels    # batch FP %
        fn = torch.count_nonzero((label_classification == 0) & (output_classification == 1)).item() / num_pixels    # batch FN %
        tn = torch.count_nonzero((label_classification == 0) & (output_classification == 0)).item() / num_pixels    # batch TN %

        # metrics updatd per batch as in original implementation (imprecise)
        metrics_meters["mse"].update(mse, 1)
        metrics_meters["baseline_mse"].update(zero_model_mse, 1)
        metrics_meters["mae"].update(mae, 1)
        metrics_meters["mave"].update(mave, 1)

        metrics_meters["tp"].update(tp, 1)
        metrics_meters["fp"].update(fp, 1)
        metrics_meters["fn"].update(fn, 1)
        metrics_meters["tn"].update(tn, 1)

        return metrics_meters

    def get_final_metrics_values(self, metrics_meters):

        # synchronize metrics across devices
        for metric_name in metrics_meters:
            metrics_meters[metric_name].all_reduce()

        tp = metrics_meters["tp"].avg
        fp = metrics_meters["fp"].avg
        fn = metrics_meters["fn"].avg
        tn = metrics_meters["tn"].avg

        if self.RANK == 0:
            print("avg TP:", tp)
            print("avg FP:", fp)
            print("avg FN:", fn)
            print("avg TN:", tn)

        # computes composite metrics from tp/fp/fn averages
        acc = (tp + tn) / (tp + fp + fn + tn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        final_metrics = {
            "test/mse": metrics_meters["mse"].avg,
            "test/baseline_mse": metrics_meters["baseline_mse"].avg,
            "test/mae": metrics_meters["mae"].avg,
            "test/mave": metrics_meters["mave"].avg,

            "test/acc": acc,
            "test/precision": precision,
            "test/recall": recall,
            "test/f1": f1,
        }

        return final_metrics

    def t_loop(self, epoch, s):

        # set model to train mode
        self.model.train()

        # Initialize the training loss meter
        train_loss_meter = AverageMeter('train_loss', ':.4f', Summary.AVERAGE)

        # Initialize the progress bar for training
        if self.RANK == 0:
            train_pbar = tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.epochs}")

        # loop training through batches
        for i, (images, labels) in enumerate(self.train_loader):
            # Move inputs and targets to the device (GPU)
            images, labels = images.to(self.device), labels.to(self.device)
            #print(f"--- DEBUGGING: TRAIN IMAGES AND LABELS DTYPE: {images.dtype} - {labels.dtype}")
            #print(f"--- DEBUGGING: TRAIN IMAGES AND LABELS min/max: {images.min()} - {images.max()} "
            #      f"- {labels.min()} - {labels.max()}")

            # Zero the gradients
            self.optimizer.zero_grad()

            # get loss
            with autocast(dtype=torch.float16):
                loss = self.get_loss(images, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # update local loss meter
            train_loss_meter.update(loss.item(), 1)

            # # Update the scheduler
            if self.lr_scheduler == 'cosine_annealing':
                s.step()

            # display progress on console
            if self.RANK == 0:
                train_pbar.update(1)

        # Close the progress bar
        if self.RANK == 0:
            train_pbar.close()
            # print("LR:", self.optimizer.param_groups[0]['lr'])

        # synchronize the epoch's losses across devices
        train_loss_meter.all_reduce()

        return train_loss_meter.avg

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
                #print(f"--- DEBUGGING: VAL IMAGES AND LABELS DTYPE: {images.dtype} - {labels.dtype}")
                #print(f"--- DEBUGGING: TRAIN IMAGES AND LABELS min/max: {images.min()} - {images.max()} "
                #      f"- {labels.min()} - {labels.max()}")

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
                # print("LR:", self.optimizer.param_groups[0]['lr'])

            if self.visualise_validation and self.RANK == 0:
                outputs = self.model(images)

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

    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize(
            x=images,
            y=labels,
            y_pred=outputs,
            images=5,
            channel_first=True,
            vmin=0,
            vmax=1,
            save_path=f"{self.out_folder}/{name}.png"
        )

    def save_ckpt(self, epoch, val_loss):
        # all metrics must be updated for early stopping to work, but only one save operation is necessary
        model_sd = self.model.state_dict().copy()

        if self.best_loss is None or val_loss < self.best_loss - self.es_delta:
            self.best_epoch = epoch
            self.best_loss = val_loss
            self.epochs_no_improve = 0 # early stopping starts counting again 
            if self.RANK == 0:
                torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_best.pt"))
            self.best_sd = model_sd

        else:
            self.epochs_no_improve += 1

        if self.RANK == 0:
            torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_last.pt"))

    def plot_curves(self, epoch):
        # visualize loss & lr curves
        self.e.append(epoch)

        fig = plt.figure()
        plt.plot(self.e, self.tl, label='Training Loss', )
        plt.plot(self.e, self.vl, label='Validation Loss')
        plt.legend()
        plt.savefig(os.path.join(self.out_folder, f"loss.png"))
        plt.close('all')
        fig = plt.figure()
        plt.plot(self.e, self.lr, label='Learning Rate')
        plt.legend()
        plt.savefig(os.path.join(self.out_folder, f"lr.png"))
        plt.close('all')

    def train(self):
        self.print_distributed("Starting distributed training...\n")

        # init model
        self.model.to(self.device)

        s = self.scheduler

        # Training loop
        for epoch in range(self.epochs):

            if self.warmup_epochs > 0:
                if epoch == 0:
                    s = self.scheduler_warmup
                    self.print_distributed('Starting linear warmup phase!')
                elif epoch == self.warmup_epochs:
                    s = self.scheduler
                    self.print_distributed('Warmup finished!')

            train_loss = self.t_loop(epoch, s)
            val_loss = self.v_loop(epoch)

            self.tl.append(train_loss)
            self.vl.append(val_loss)
            self.lr.append(self.optimizer.param_groups[0]['lr'])

            # Update the scheduler
            if epoch < self.warmup_epochs:
                s.step()
            elif self.lr_scheduler == 'reduce_on_plateau':
                s.step(self.vl[-1])

            # save checkpoint and update early stopping metrics with all_reduced val_loss
            # must update es metrics on all processes to guarantee they all stop at the same iteration
            self.save_ckpt(epoch, val_loss)

            if self.RANK == 0:

                # visualize loss & lr curves
                self.plot_curves(epoch)

                # Logging metrics on wandb
                self.wandb.log({
                    "train/loss": train_loss,
                    "val/loss": val_loss,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "epoch": epoch,
                })

            # Early stopping, for all processes
            if self.epochs_no_improve == self.es_patience:
                self.print_distributed(f'Early stopping triggered after {epoch + 1} epochs.')
                self.last_epoch = epoch + 1
                break

    def test(self):

        self.print_distributed(f"Finished Training. Best epoch: {self.best_epoch + 1}\n")
        self.print_distributed("Starting Testing...")

        # Load the best weights
        self.model.load_state_dict(self.best_sd)

        # set model to evaluation mode
        self.model.eval()

        # setup distributed metrics meters
        metrics_meters = self.get_metrics_meters()

        if self.RANK == 0:
            test_pbar = tqdm(total=len(self.test_loader), desc=f"Test Set")

        with torch.no_grad():
            for k, (images, labels) in enumerate(self.test_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)
                #print(f"--- DEBUGGING: TEST IMAGES AND LABELS DTYPE: {images.dtype} - {labels.dtype}")
                #print(f"--- DEBUGGING: TRAIN IMAGES AND LABELS min/max: {images.min()} - {images.max()} "
                #      f"- {labels.min()} - {labels.max()}")

                # update local metrics meters
                metrics_meters = self.update_metrics_meters(metrics_meters, labels, images)

                if self.RANK == 0:
                    test_pbar.update(1)

            # synchronize the metrics across devices
            self.test_metrics = self.get_final_metrics_values(metrics_meters)

            if self.RANK == 0:

                # Close the progress bar
                test_pbar.close()

                # Print test metrics
                print(f"Test Metrics:")
                for metric in self.test_metrics:
                    print(f"{metric}: {self.test_metrics[metric]}")

                # Log test metrics on wandb
                self.wandb.log(self.test_metrics)

                outputs = self.model(images)
                # Visualize prediction examples
                self.val_visualize(
                    images=images.detach().cpu().numpy(),
                    labels=labels.detach().cpu().numpy(),
                    outputs=outputs.detach().cpu().numpy(),
                    name='test'
                )

                # save the final model checkpoint
                model_sd = self.model.state_dict().copy()
                torch.save(model_sd, os.path.join(self.out_folder, f"{self.name}_final.pt"))

    # TODO restore model_summary
    def save_info(self, model_summary=None, n_shot=None, p_split=None, warmup_epochs=None, lr=None):
        if self.RANK == 0:
            artifacts = {
                'training_parameters': {
                    'model': self.name,
                    'lr': lr,
                    'scheduler': self.lr_scheduler,
                    'warmup_epochs': warmup_epochs,
                    'optimizer': str(self.optimizer).split(' (')[0],
                    'device': str(self.device),
                    'training_epochs': self.epochs,
                    'es_patience': self.es_patience,
                    'es_delta': self.es_delta,
                    #'train_samples': len(self.train_loader) * model_summary.input_size[0][0],
                    #'val_samples': len(self.val_loader) * model_summary.input_size[0][0],
                    #'test_samples': len(self.test_loader) * model_summary.input_size[0][0],
                    'n_shot': n_shot,
                    'p_split': p_split
                },

                'training_info': {
                    'best_val_loss': self.best_loss,
                    'best_epoch': self.best_epoch,
                    'last_epoch': self.last_epoch
                },

                'test_metrics': self.test_metrics,

                'plot_info': {
                    'epochs': self.e,
                    'val_losses': self.vl,
                    'train_losses': self.tl,
                    'lr': self.lr
                },

                #'model_summary': {
                #    'batch_size': model_summary.input_size[0],
                #    'input_size': model_summary.total_input,
                #    'total_mult_adds': model_summary.total_mult_adds,
                #    'back_forward_pass_size': model_summary.total_output_bytes,
                #    'param_bytes': model_summary.total_param_bytes,
                #    'trainable_params': model_summary.trainable_params,
                #    'non-trainable_params': model_summary.total_params - model_summary.trainable_params,
                #    'total_params': model_summary.total_params
                #}
            }

            with open(f"{self.out_folder}/artifacts.json", "w") as outfile:
                json.dump(artifacts, outfile)
