"""
Implementation of SN_MRF_CC model adapted from:

Jiangxin Yang, Guizhong Fu, Wenbin Zhu, Yanlong Cao, Yanpeng Cao, and Michael Ying Yang.  
"A deep learning-based surface defect inspection system using multiscale and channel-compressed features."  
IEEE Transactions on Instrumentation and Measurement, 69(10):8032–8042, 2020.

This model combines multi-receptive field (MRF) branches with a channel-compressed shortcut connection, 
as described in the original paper.
"""


import torch
import torch.nn as nn
from model import SqueezeNet


class SN_MRF_CC(nn.Module):
    def __init__(self, num_classes=7, Cn=6):
        super().__init__()
        self.squeezenet = SqueezeNet()
        self.Cn = Cn

        self.MRF_a = nn.Sequential(nn.Conv2d(512, self.Cn, kernel_size=1, padding=0), nn.ReLU())
        self.MRF_b = nn.Sequential(nn.Conv2d(512, self.Cn, kernel_size=3, padding=1), nn.ReLU())
        self.MRF_c = nn.Sequential(nn.Conv2d(512, self.Cn, kernel_size=5, padding=2), nn.ReLU())

        # Short connection + fusion → Pool10
        self.pool10 = nn.AdaptiveAvgPool2d((1, 1))  # [B, 3*Cn, 1, 1]
        self.conv10 = nn.Conv2d(3 * self.Cn, num_classes, kernel_size=1)

        self.short_conv = nn.Conv2d(512, 3 * self.Cn, kernel_size=1)

    def forward(self, images):
        features = self.squeezenet(images)  # output: [B, 512, H, W] (Fire9/concat)

        # 2. MRF branches
        out_a = self.MRF_a(features)  # [B, Cn, H, W]
        out_b = self.MRF_b(features)  # [B, Cn, H, W]
        out_c = self.MRF_c(features)  # [B, Cn, H, W]

        # 3. Concatenate multi-scale features
        fused = torch.cat([out_a, out_b, out_c], dim=1)  # [B, 3*Cn, H, W]

        # 4. Short connection
        shortcut = self.short_conv(features)  # [B, 3*Cn, H, W]
        residual = fused + shortcut  # [B, 3*Cn, H, W]

        # 5. Pooling & classification
        pooled = self.pool10(residual)  # [B, 3*Cn, 1, 1]
        logits = self.conv10(pooled).squeeze(-1).squeeze(-1)  # [B, num_classes]

        return logits