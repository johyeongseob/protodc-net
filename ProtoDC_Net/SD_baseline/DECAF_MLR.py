"""
Reference:
Ren, Ruoxu, Terence Hung, and Kay Chen Tan.
"A generic deep-learning-based approach for automated surface inspection."
IEEE Transactions on Cybernetics 48.3 (2017): 929-940.

Note:
In the original paper, a pretrained Decaf model was used as a feature extractor.
In this implementation, we replace Decaf with a pretrained AlexNet from PyTorch,
which has a similar architecture. The fc6 layer is used as the transferred feature,
and the pretrained weights are frozen during training.
"""


import torch
import torch.nn as nn
import torchvision.models as models

class AlexNetFC6Extractor(nn.Module):
    def __init__(self):
        super().__init__()

        # 1. Pretrained AlexNet 로드
        alexnet = models.alexnet(pretrained=True)

        # 2. 전체 파라미터 freeze
        for param in alexnet.parameters():
            param.requires_grad = False

        # 3. 필요한 부분만 구성
        self.features = alexnet.features  # Conv1~5
        self.avgpool = alexnet.avgpool
        self.fc6 = alexnet.classifier[0]
        self.relu6 = alexnet.classifier[1]
        self.dropout6 = alexnet.classifier[2]

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc6(x)
        x = self.relu6(x)
        x = self.dropout6(x)
        return x  # shape: [batch, 4096]

class DECAF_MLR(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()
        self.feature_extractor = AlexNetFC6Extractor()
        self.classifier = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)   # fc6 feature (4096-dim)
        x = self.classifier(x)          # classification logits
        return x
