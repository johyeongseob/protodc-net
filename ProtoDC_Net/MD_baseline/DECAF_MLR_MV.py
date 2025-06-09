"""
Multi-view extension of the DECAF-MLR model.

This model processes multiple images of the same sample acquired under different illumination 
conditions or augmentations. Each view is passed independently through the shared feature extractor 
(SqueezeNet), followed by average pooling and multinomial logistic regression. 
The final prediction is obtained by averaging the logits across all views.

Based on:
Ren, Ruoxu, Terence Hung, and Kay Chen Tan. 
"A generic deep-learning-based approach for automated surface inspection." 
IEEE Transactions on Cybernetics 48.3 (2017): 929–940.
"""


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.nn.functional as F
from SD_baseline.DECAF_MLR import DECAF_MLR

class DECAF_MLR_MV(nn.Module):
    def __init__(self, num_classes=5):
        super(DECAF_MLR_MV, self).__init__()

        # features 부분만 사용 (classifier 제외)
        self.DECAF_MLR = DECAF_MLR()

    def forward(self, images):
        """
        images: [B, 4, 3, 200, 200] (Batch, Views, Channels, Height, Width)
        """
        B, V, C, H, W = images.size()  # B: Batch, V: Views
        assert V == 4, "This model is designed for 4 views only."

        output_stack = torch.stack([self.DECAF_MLR(images[:, v]) for v in range(V)], dim=0)  # [4, B, 5]
        averaged_logit = torch.mean(output_stack, dim=0)  # [B, 5]

        return averaged_logit
