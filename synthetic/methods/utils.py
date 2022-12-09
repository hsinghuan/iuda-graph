import torch

def negative_entropy_from_logits(logits):
    neg_entropies = torch.sum(
        torch.softmax(logits, dim=1) * torch.log_softmax(logits, dim=1), dim=1
    )
    return torch.mean(neg_entropies)