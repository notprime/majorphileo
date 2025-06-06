"""
Training utils

1. WeightClipper;
2. Weight initialization for SELU activation
3. Orthogonal weight initialization
"""

import numpy as np

import torch


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
            path (str): Path for the checkpoint to be saved to. Default: 'checkpoint.pt'
            trace_func (function): trace print function. Default: print
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, rank=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, rank)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, rank)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, rank=None):
        """ Saves model when validation loss decrease. """

        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        if rank is not None:
            ckp = model.module.state_dict()
            torch.save(ckp, self.path)
        else:
            torch.save(model.state_dict(), self.path)

        self.val_loss_min = val_loss
