import math
import random

import numpy as np
import torch


def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=0, start_warmup_value=0, warmup_steps=-1):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs
    if warmup_steps > 0:
        warmup_iters = warmup_steps

    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs - warmup_iters)
    schedule = np.array(
        [final_value + 0.5 * (base_value - final_value) * (1 + math.cos(math.pi * i / (len(iters)))) for i in iters]
    )

    schedule = np.concatenate((warmup_schedule, schedule))

    assert len(schedule) == epochs

    return schedule


class RandomChannelDropout(object):
    """ Randomly drops one of the channels of the input image with a given probability. """

    def __init__(self, p=0.5):
        """
        Args:
            p (float): probability of the image being transformed. Default value is 0.5.
        """

        super(RandomChannelDropout, self).__init__()
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (Tensor): Image to be transformed.

        Returns:
            Tensor: Transformed image with one channel potentially dropped.
        """

        # Ensure image is in the form of a PyTorch Tensor
        if not isinstance(img, torch.Tensor):
            raise TypeError('Input img should be a torch.Tensor but got type {}'.format(type(img)))

        # Randomly decide whether to drop a channel based on p
        if random.random() < self.p:
            # Randomly select a channel index (0 for Red, 1 for Green, 2 for Blue)
            channel_to_drop = random.randint(0, img.shape[0] - 1)

            # Set the selected channel to zero
            img[channel_to_drop, :, :] = 0

        return img
