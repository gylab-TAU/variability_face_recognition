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

        # compute triplet loss
        # logits = sim_pos - sim_neg
        # labels = torch.ones(logits.size(0), dtype=torch.long).to(anchor.device)


        loss = nn.CrossEntropyLoss()(logits, labels)
        return loss