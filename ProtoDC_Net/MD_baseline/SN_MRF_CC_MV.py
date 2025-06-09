"""
Multi-view extension of the SN_MRF_CC model based on:

Jiangxin Yang, Guizhong Fu, Wenbin Zhu, Yanlong Cao, Yanpeng Cao, and Michael Ying Yang.  
"A deep learning-based surface defect inspection system using multiscale and channel-compressed features."  
IEEE Transactions on Instrumentation and Measurement, 69(10):8032–8042, 2020.

This model processes four views of the same sample using a shared SN_MRF_CC module.  
Each view is passed independently through the base model, and the final prediction is obtained  
by averaging the classification logits across views.
"""


import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn as nn
import torch.nn.functional as F
from SD_baseline.SN_MRF_CC import SN_MRF_CC

class SN_MRF_CC_MV(nn.Module):
    def __init__(self, num_classes=5):
        super(SN_MRF_CC_MV, self).__init__()

        self.SN_MRF_CC = SN_MRF_CC()

    def forward(self, images):
        """
        images: [B, 4, 3, 200, 200] (Batch, Views, Channels, Height, Width)
        """
        B, V, C, H, W = images.size()  # B: Batch, V: Views
        assert V == 4, "This model is designed for 4 views only."

        outputs_stack = torch.stack([self.SN_MRF_CC(images[:, v]) for v in range(V)], dim=0)  # [4, B, 5]
        averaged_logits = torch.mean(outputs_stack, dim=0)  # [B, 5]

        return averaged_logits
