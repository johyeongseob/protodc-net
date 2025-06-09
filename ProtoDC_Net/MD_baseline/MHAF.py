"""
Implementation of MHAF (Multi-Head Attention Fusion) model adapted from:¹

Michele Somero, Federico Urli, Lauro Snidaro, and Alessandro Liani.  
"Defect detection multiheadattention fusion model on images acquired with different light sources."  
In Proceedings of the 2024 27th International Conference on Information Fusion (FUSION), pages 1–6. IEEE, 2024.

This model applies multi-head attention to fuse feature maps from multi-illumination inputs, 
following the approach described in the paper.
"""


import torch, copy
import torch.nn.functional as F
import torch.nn as nn
from model import SqueezeNet, SENet

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.3):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm([13 * 13, embed_dim])
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k):
        B, C, H, W = q.size()
        q = q.view(B, C, H * W).permute(0, 2, 1)  # (B, 13*13, 512)
        k = k.view(B, C, H * W).permute(0, 2, 1)  # (B, 13*13, 512)

        out, _ = self.mha(q, k, k)
        out = self.norm(out)
        out = self.dropout(out)
        return out.permute(0, 2, 1).view(B, C, H, W)

class MHAF(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.views = 4
        self.squeezenet_views = nn.ModuleList([copy.deepcopy(SqueezeNet()) for _ in range(self.views)])
        self.concatconv_views = nn.ModuleList([
            copy.deepcopy(
                nn.Sequential(
                    nn.Conv2d(in_channels=512*3, out_channels=512, kernel_size=1),
                    nn.ReLU()
                )
            ) for _ in range(4)
        ])
        self.attn_blocks = nn.ModuleList([MultiHeadAttentionBlock() for _ in range(self.views)])
        self.concat_dropout = nn.Dropout(0.1)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels=512 * 4, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1),
            nn.ReLU()
        )


    def forward(self, images):
        """
        images: [B, 4, 3, 224, 224] (Batch, Views, Channels, Height, Width)
        """
        B, V, C, H, W = images.size()  # B: Batch, V: Views
        assert V == len(self.squeezenet_views), "This model is designed for 4 views only."

        features = []
        for v in range(V):
            view_images = images[:, v, :, :, :]  # [B, 3, H, W]
            feature = self.squeezenet_views[v](view_images)
            features.append(feature)

        mha_outputs = []
        for i in range(V):
            others = torch.cat([features[j] for j in range(V) if j != i], dim=1)  # (B, 512*3, 13, 13)
            others = self.concatconv_views[i](others)
            mha_out = self.attn_blocks[i](features[i], others)
            mha_outputs.append(mha_out)

        fused = torch.cat(mha_outputs, dim=1)  # (B, 2048, 13, 13)
        fused = self.concat_dropout(fused)  # (B, 2048, 13, 13)
        fused = self.fusion_conv(fused)  # (B, 5, 13, 13)
        logit = F.adaptive_avg_pool2d(fused, (1, 1)).view(B, -1)  # [B, 5]

        return logit
