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

    def LFM(self, features_lists):
        B = features_lists[0].size(0)
        embedded_list = [F.adaptive_avg_pool2d(f, (1, 1)).view(B, 1, -1) for f in features_lists]
        local_stack = torch.cat(embedded_list, dim=1)  # [B, 4, C]
        local_vector = torch.median(local_stack, dim=1).values  # [B, C]
        return local_vector

    def forward(self, images):
        """
        images: [B, 4, 3, 200, 200] (Batch, Views, Channels, Height, Width)
        """
        B, V, C, H, W = images.size()  # B: Batch, V: Views
        assert V == 4, "This model is designed for 5 views only. 4 are defect view, 1 is normal view"

        features_lists = [self.senet_views[v](self.squeezenet_views[v](images[:, v, :, :, :])) for v in range(V)]

        return self.LFM(features_lists)


class ProtoDC_local(nn.Module):
    def __init__(self):
        super(ProtoDC_local, self).__init__()
        self.EmbeddingInput = EmbeddingInput()
        self.log_temperature = nn.Parameter(torch.tensor(np.log(1)))

    def forward(self, views, target):
        """
        views: [B, 4, 3, H, W]
        labels: [B]
        """
        B = views.size(0)

        local_vectors = self.EmbeddingInput(views)  # [B, C], [B, C]

        # Normalize (cosine similarity 용)
        local_norm = F.normalize(local_vectors, dim=1)  # [B, C]
        
        # global → global
        logits_ll = torch.matmul(local_norm, local_norm.T)  # [B, B]
        logits_ll = logits_ll / self.log_temperature.exp()

        # 자기 자신은 제외
        diag_mask = torch.eye(B, device=views.device).bool()
        logits_ll.masked_fill_(diag_mask, -1e9)

        # label_matrix: [B, B], 같은 클래스 = 1
        label_matrix = (target.unsqueeze(1) == target.unsqueeze(0)).float()

        label_matrix_ll = label_matrix.clone()
        label_matrix_ll.fill_diagonal_(0)

        log_prob_ll = F.log_softmax(logits_ll, dim=1)
        loss_ll = -(label_matrix_ll * log_prob_ll).sum(1) / (label_matrix_ll.sum(1) + 1e-9)

        return loss_ll.mean()


if __name__ == '__main__':
    x = torch.rand(8, 4, 3, 200, 200)
    labels = torch.Tensor([0, 0, 0, 1, 1, 1, 2, 2])
    Classifier = SupCLIPLoss()

    out = Classifier(x, labels)
    print(f"out: {out}")