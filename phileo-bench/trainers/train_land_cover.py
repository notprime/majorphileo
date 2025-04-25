# PyTorch
import torch
import torch.nn as nn
import numpy as np

# utils
from utils import visualize, config_lc
from utils.metrics import Summary, AverageMeter

# trainers
from trainers.train_base import TrainBase


class TrainLandCover(TrainBase):

    def set_criterion(self):
        return nn.CrossEntropyLoss()

    def get_loss(self, images, labels):
        outputs = self.model(images)
        outputs = outputs.flatten(start_dim=2).squeeze()
        labels = labels.flatten(start_dim=1).squeeze()
        loss = self.criterion(outputs, labels)
        return loss

    def val_visualize(self, images, labels, outputs, name):
        visualize.visualize_lc(
            x=images,
            y=labels,
            y_pred=outputs.argmax(axis=1),
            images=5,
            channel_first=True,
            vmin=0,
            save_path=f"{self.out_folder}/{name}.png"
        )

    def get_metrics_meters(self):
        num_classes = len(config_lc.lc_raw_classes.keys())
        metrics_meters = {}

        for c1 in range(num_classes):
            for c2 in range(num_classes):
                metrics_meters[f"{c1}_{c2}"] = AverageMeter(f'{c1}_{c2}', ':.4f', Summary.AVERAGE)

        return metrics_meters

    def update_metrics_meters(self, metrics_meters, labels, images):
        outputs = self.model(images)
        outputs = outputs.argmax(axis=1).flatten()
        labels = labels.squeeze().flatten()

        # stolen from pytorch confusion matrix
        num_classes = len(config_lc.lc_raw_classes.keys())
        unique_mapping = labels.to(torch.long) * num_classes + outputs.to(torch.long)
        bins = torch.bincount(unique_mapping, minlength=num_classes ** 2)
        cfm = bins.reshape(num_classes, num_classes)
        cfm = cfm.cpu().numpy()

        for c1 in range(num_classes):
            for c2 in range(num_classes):
                metrics_meters[f"{c1}_{c2}"].update(cfm[c1, c2], 1)

        return metrics_meters

    def get_final_metrics_values(self, metrics_meters):
        num_classes = len(config_lc.lc_raw_classes.keys())
        confmat = np.zeros((num_classes, num_classes))

        for c1 in range(num_classes):
            for c2 in range(num_classes):
                metrics_meters[f"{c1}_{c2}"].all_reduce()
                confmat[c1, c2] = metrics_meters[f"{c1}_{c2}"].sum

        total_pixels = np.sum(confmat)

        tp_per_class = np.diagonal(confmat)
        total_tp = tp_per_class.sum()

        fp_per_class = confmat.sum(axis=0) - tp_per_class
        fn_per_class = confmat.sum(axis=1) - tp_per_class

        precision_per_class = tp_per_class / (fp_per_class + tp_per_class)
        recall_per_class = tp_per_class / (fn_per_class + tp_per_class)

        precision_micro = total_tp / (fp_per_class.sum() + total_tp)
        recall_micro = total_tp / (fn_per_class.sum() + total_tp)
        precision_macro = np.mean(precision_per_class)
        recall_macro = np.mean(recall_per_class)

        acc_total = total_tp / total_pixels

        final_metrics = {
            'acc': acc_total,
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'precision_micro': precision_micro,
            'precision_macro': precision_macro,
            'recall_micro': recall_micro,
            'recall_macro': recall_macro,
            'conf_mat': confmat.tolist()
        }

        return final_metrics
