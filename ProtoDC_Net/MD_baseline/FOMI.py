"""
FOMI (Fusion of Multi-Illumination) model based on:

Guizhong Fu, Shukai Jia, Wenbin Zhu, Jiangxin Yang, Yanlong Cao, Michael Ying Yang, and Yanpeng Cao.  
"Fusion of multi-light source illuminated images for effective defect inspection on highly reflective surfaces."  
Mechanical Systems and Signal Processing, 175:109109, 2022.

This model processes four illumination-specific views using SqueezeNet and SENet backbones.  
Each view produces individual classification logits and feature maps.  
The features are then concatenated and passed through an additional SENet and fusion layers to produce the final prediction.
"""


import torch, copy
import torch.nn.functional as F
import torch.nn as nn
from model import SqueezeNet, SENet


class FOMI(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.squeezenet = SqueezeNet()
        self.senet = SENet(c=512)
        self.conv10 = nn.Sequential(
                    nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1),
                    nn.ReLU()
                )
        self.fused_senet = SENet(c=num_classes*4)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels=num_classes*4, out_channels=num_classes, kernel_size=3, padding=1),
        )

    def process(self, view_images):
        features = self.squeezenet(view_images)  # [B, C, H', W']
        features = self.senet(features)         # [B, 512, H', W']
        feature = self.conv10(features)        # [B, 5, H', W']
        logit = F.adaptive_avg_pool2d(feature, (1, 1)).view(feature.size(0), -1)  # [B, 5]
        return feature, logit

    def forward(self, images):
        """
        images: [B, 4, 3, 200, 200] (Batch, Views, Channels, Height, Width)
        """
        B, V, C, H, W = images.size()  # B: Batch, V: Views
        assert V == 4, "This model is designed for 4 views only."

        features, logits = [], []
        for v in range(V):
            feature, logit = self.process(images[:, v, :, :, :])
            features.append(feature)
            logits.append(logit)

        fused_features = torch.cat(features, dim=1)  # [B, 5*4, H', W']
        fused_features = self.fused_senet(fused_features)  # [B, 5*4, H', W']
        fused_features = self.fusion_conv(fused_features)  # [B, 5, H', W']
        fused_logit = F.adaptive_avg_pool2d(fused_features, (1,1)).view(B, -1)  # [B, 5]

        return logits, fused_logit
