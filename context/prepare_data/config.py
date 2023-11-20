import os
import numpy as np
from dataclasses import dataclass

@dataclass
class TrainSetConfig():
    num_classes: int = 100
    train_num_instances_per_class: int = 50
    val_num_instances_per_class: int = 10
    test_num_instances_per_class: int = 1
    num_classes_per_category: int = 10

    data_dir: str = r"C:\Users\shiri\Documents\School\Master\Research\Face_recognition\Data\VGG-Face2\MTCNN\train_mtcnn_aligned_2000"
    root_dir: str = r"C:\Users\shiri\Documents\School\Master\Research\Context"