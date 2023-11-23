import os
import numpy as np
from dataclasses import dataclass

@dataclass
class TrainConfig():
    exp_name: str = 'test'
    num_epoch_contrastive: int = 1
    num_epoch_fine_tune: int = 1
    num_classes: int = 4
    batch_size: int = 8

