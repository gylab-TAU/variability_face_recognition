import os
import torch
from torch import nn

def categorical_accuracy(pred_category, label_category, category_id_dict):
    '''
    This function calculates the categorical accuracy of a single prediction.
    '''
    acc = 0
    # convert label_category[i] to int
    pred_category = [int(i) for i in pred_category]
    for i in range(len(label_category)):
        if category_id_dict[pred_category[i]] == label_category[i]:
            acc += 1
    return acc / len(label_category)


# create a categorical accuracy loss function where all elements are tensors
def categorical_accuracy_loss(pred_category, label_category, category_id_dict):
    '''
    This function calculates the categorical accuracy of a single prediction.
    '''
    acc = torch.tensor(0)
    # convert label_category[i] to int
    pred_category = [int(i) for i in pred_category]
    for i in range(len(label_category)):
        if category_id_dict[pred_category[i]] == label_category[i]:
            acc += 1
    return acc / len(label_category)

class CategoricalAccLoss(nn.modules.loss._Loss):
    def __init__(self, category_id_dict):
        super().__init__()
        self.category_id_dict = category_id_dict

    def forward(self, pred_category, label_category):
        acc = torch.tensor(0)
        # convert label_category[i] to int
        pred_category = [int(i) for i in pred_category]
        for i in range(len(label_category)):
            if self.category_id_dict[pred_category[i]] == label_category[i]:
                acc += 1
        # convet to float
        acc = acc.type(torch.FloatTensor)
        return torch.mean(acc)

class DummyLoss(nn.modules.loss._Loss):
    def __init__(self):
        super().__init__()

    def forward(self, pred_category, label_category):
        return torch.mean(pred_category)

# class CategoricalCrossEntropy(nn.modules.loss._Loss):
#     def __init__(self, category_id_dict):
#         super().__init__()
#         self.category_id_dict = category_id_dict
#
#     def forward(self, pred_category, label_category):
#         # convert label_category[i] to int
#         pred_category = [int(i) for i in pred_category]
#         for i in range(len(label_category)):
#             if self.category_id_dict[pred_category[i]] == label_category[i]:
#                 acc += 1
#         # convet to float
#         acc = acc.type(torch.FloatTensor)
#         return torch.mean(acc