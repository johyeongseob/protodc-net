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
        self.fused_senet = SENet(c=512 * 4)
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels=512 * 4, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def LFM(self, features_lists):
        B = features_lists[0].size(0)
        embedded_list = [F.adaptive_avg_pool2d(f, (1, 1)).view(B, 1, -1) for f in features_lists]
        local_stack = torch.cat(embedded_list, dim=1)  # [B, 4, C]
        local_vector = torch.median(local_stack, dim=1).values  # [B, C]
        return local_vector

    def GFM(self, features_lists):
        fusion = torch.cat(features_lists, dim=1)  # [B, C*4, H', W']
        global_feature = self.fusion_conv(self.fused_senet(fusion))
        global_vector = F.adaptive_avg_pool2d(global_feature, (1, 1))  # [B, C, 1, 1]
        return global_vector.squeeze(3).squeeze(2)

    def forward(self, images):
        """
        images: [B, 4, 3, 200, 200] (Batch, Views, Channels, Height, Width)
        """
        B, V, C, H, W = images.size()  # B: Batch, V: Views
        assert V == 4, "This model is designed for 5 views only. 4 are defect view, 1 is normal view"

        features_lists = [self.senet_views[v](self.squeezenet_views[v](images[:, v, :, :, :])) for v in range(V)]

        return self.LFM(features_lists), self.GFM(features_lists)


class cosine_compactness_loss(nn.Module):
    def __init__(self):
        super(cosine_compactness_loss, self).__init__()
        self.EmbeddingInput = EmbeddingInput()

    def forward(self, views):
        """
        views: [B, 4, 3, H, W]
        labels: [B]
        """

        local_vec, global_vec = self.EmbeddingInput(views)  # [B, C], [B, C]

        local_vec = F.normalize(local_vec, dim=1)
        global_vec = F.normalize(global_vec, dim=1)

        mu_l = local_vec.mean(dim=0, keepdim=True)
        mu_g = global_vec.mean(dim=0, keepdim=True)  # [1, C]

        sim_l = F.cosine_similarity(local_vec, mu_l, dim=1)  # [B]
        sim_g = F.cosine_similarity(global_vec, mu_g, dim=1)  # [B]

        loss_l = ((1 - sim_l) ** 2).mean()
        loss_g = ((1 - sim_g) ** 2).mean()
        return (loss_l + loss_g) / 2


if __name__ == '__main__':
    x = torch.rand(8, 4, 3, 200, 200)
    labels = torch.Tensor([0, 0, 0, 1, 1, 1, 2, 2])
    # Classifier = SupCLIPLoss()
