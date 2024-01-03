import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negative):
        # Calculate cosine similarity between anchor and positive/negative samples
        sim_pos = torch.cosine_similarity(anchor, positive, dim=-1) / self.temperature
        sim_neg = torch.cosine_similarity(anchor, negative, dim=-1) / self.temperature

        # Compute contrastive loss using the InfoNCE (Noise Contrastive Estimation) loss formula
        logits = torch.cat([sim_pos.unsqueeze(1), sim_neg.unsqueeze(1)], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(anchor.device)



        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def calc_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()