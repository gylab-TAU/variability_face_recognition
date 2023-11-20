import os
import numpy as np
from dataclasses import dataclass

@dataclass
class TrainConfig():
    exp_name: str = 'test'
    num_epoch_contrastive: int = 10
    num_epoch_fine_tune: int = 10
    batch_size: int = 8

