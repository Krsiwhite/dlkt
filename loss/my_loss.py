import torch.nn as nn
import torch.nn.functional as F

class SmoothBCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, pred, target):
        # target: [0, 1] -> [0.05, 0.95]
        target = target * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = F.binary_cross_entropy(pred, target)
        return loss