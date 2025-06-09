import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from model import SqueezeNet, SENet
import numpy as np



class EmbeddingInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.squeezenet_views = nn.ModuleList([copy.deepcopy(SqueezeNet()) for _ in range(4)])
        self.senet_views = nn.ModuleList([copy.deepcopy(SENet(c=512)) for _ in range(4)])
        self.fusion_senet = SENet(c=512*4)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels=512 * 4, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def GFM(self, features_lists):
        fusion = torch.cat(features_lists, dim=1)  # [B, C*4, H', W']
        global_feature = self.fusion_conv(self.fused_senet(fusion))
        global_vector = F.adaptive_avg_pool2d(global_feature, (1, 1))  # [B, C, 1, 1]
        return global_vector

    def forward(self, images):
        """
        images: [B, 4, 3, 200, 200] (Batch, Views, Channels, Height, Width)
        """
        B, V, C, H, W = images.size()  # B: Batch, V: Views
        assert V == 4, "This model is designed for 5 views only. 4 are defect view, 1 is normal view"

        features_list = [self.senet_views[v](self.squeezenet_views[v](images[:, v, :, :, :])) for v in range(V)]

        return self.GFM(features_list)


class ProtoDC_global(nn.Module):
    def __init__(self):
        super(ProtoDC_global, self).__init__()
        self.EmbeddingInput = EmbeddingInput()
        self.log_temperature = nn.Parameter(torch.tensor(np.log(1)))

    def forward(self, views, target):
        """
        views: [B, 4, 3, H, W]
        labels: [B]
        """
        B = views.size(0)

        global_vectors = self.EmbeddingInput(views)  # [B, C], [B, C]

        # Normalize (cosine similarity 용)
        global_norm = F.normalize(global_vectors, dim=1)  # [B, C]

        # global → global
        logits_gg = torch.matmul(global_norm, global_norm.T)  # [B, B]
        logits_gg = logits_gg / self.log_temperature.exp()

        # 자기 자신은 제외
        diag_mask = torch.eye(B, device=views.device).bool()
        logits_gg.masked_fill_(diag_mask, -1e9)

        # label_matrix: [B, B], 같은 클래스 = 1
        label_matrix = (target.unsqueeze(1) == target.unsqueeze(0)).float()

        label_matrix_gg = label_matrix.clone()
        label_matrix_gg.fill_diagonal_(0)

        log_prob_gg = F.log_softmax(logits_gg, dim=1)
        loss_gg = -(label_matrix_gg * log_prob_gg).sum(1) / (label_matrix_gg.sum(1) + 1e-9)

        return loss_gg.mean()


if __name__ == '__main__':
    x = torch.rand(8, 4, 3, 200, 200)
    labels = torch.Tensor([0, 0, 0, 1, 1, 1, 2, 2])
    Classifier = SupCLIPLoss()

    out = Classifier(x, labels)
    print(f"out: {out}")