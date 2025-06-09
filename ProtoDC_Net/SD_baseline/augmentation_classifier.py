import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch, copy
import torch.nn as nn
import torch.nn.functional as F
from model import SqueezeNet
import numpy as np



class EmbeddingInput(nn.Module):
    def __init__(self):
        super().__init__()
        self.squeezenet = SqueezeNet()
        # self.senet = SENet(c=512)

    def forward(self, images):
        """
        images: [B, 3, 200, 200] (Batch, Channels, Height, Width)
        """
        features = self.squeezenet(images)
        # features = self.senet(features)
        embedded = F.adaptive_avg_pool2d(features, (1, 1)).squeeze(3).squeeze(2)

        return embedded


class augmentation_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.squeezenet = SqueezeNet()
        self.EmbeddingInput = EmbeddingInput()
        # self.projection = nn.Linear(512, 256)
        self.log_temperature = nn.Parameter(torch.tensor(np.log(1.0)))


    def forward(self, images, targets):
        """
        images: [B, 3, 200, 200] (Batch, Channels, Height, Width)
        """

        embedded = self.EmbeddingInput(images)
        norm = F.normalize(embedded, dim=1)

        # Cosine similarity logits: [B, B]
        logits = torch.matmul(norm, norm.T)  # [B, B]
        logits = logits / self.log_temperature.exp()

        # label_matrix: [B, B], 같은 클래스 = 1
        label_matrix = (targets.unsqueeze(1) == targets.unsqueeze(0)).float()
        label_matrix.fill_diagonal_(0)

        log_prob = F.log_softmax(logits, dim=1)
        loss = -(label_matrix * log_prob).sum(1) / (label_matrix.sum(1) + 1e-9)

        return loss.mean()


if __name__ == '__main__':
    x = torch.rand(8, 3, 200, 200)
    labels = torch.Tensor([0, 0, 0, 1, 1, 1, 2, 2])
    Classifier = SupCLIPLoss_SV()

    out = Classifier(x, labels)
    print(f"out: {out}")
