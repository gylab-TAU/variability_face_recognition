import os
import numpy as np
from dataclasses import dataclass

@dataclass
class TrainConfig():
    exp_name: str = 'test'
    data_dir: str = r'C:\Users\shiri\Documents\School\Master\Research\Face_recognition\Data\VGG-Face2\MTCNN\train_mtcnn_aligned_2000'
    num_epoch_contrastive: int = 1
    num_epoch_fine_tune: int = 1
    num_classes: int = 50
    num_samples_per_class: int = 10
    batch_size: int = 8

