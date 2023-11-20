import os
import numpy as np
from dataclasses import dataclass

@dataclass
class TrainConfig():
    exp_name: str = 'test'
    exp_description: str = 'test'
    num_epochs: int = 10
    batch_size: int = 8

