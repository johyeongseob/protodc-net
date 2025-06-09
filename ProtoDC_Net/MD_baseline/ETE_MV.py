"""
Multi-view extension of the ETEClassifier model.

This model processes multiple images of the same sample acquired under different illumination conditions 
or generated through diverse data augmentations. Each view is processed independently using an ETEClassifier, 
and their outputs are fused by averaging the classification logits.
"""


import torch.nn as nn
import torch, copy
from SD_baseline.ETE import ETEClassifier

class ETE_MV(nn.Module):
    def __init__(self, num_classes=5):
        super(ETE_MV, self).__init__()

        self.ETE = ETE()

    def forward(self, images):
        """
        images: [B, 4, 3, 200, 200] (Batch, Views, Channels, Height, Width)
        """
        B, V, C, H, W = images.size()  # B: Batch, V: Views
        assert V == 4, "This model is designed for 4 views only."

        outputs_stack = torch.stack([self.ETEClassifier(images[:, v]) for v in range(V)], dim=0)  # [4, B, 5]
        averaged_logits = torch.mean(outputs_stack, dim=0)  # [B, 5]

        return averaged_logits
