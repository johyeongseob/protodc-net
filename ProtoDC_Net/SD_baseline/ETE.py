"""
End-to-end defect classification network reimplemented based on the model proposed in:¹

Li Yi, Guangyao Li, and Mingming Jiang. 
"An end-to-end steel strip surface defects recognition system based on convolutional neural networks." 
Steel Research International, 88(2):1600068, 2017.

This implementation reproduces the general structure described in the paper,
including a convolutional feature extractor and a multi-layer classifier.
"""


import torch.nn as nn

class ETE(nn.Module):
    def __init__(self, num_classes=5):
        super(ETE, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5),      # (3, 144, 144) -> (32, 140, 140)
            nn.ReLU(),
            nn.MaxPool2d(2),                      # -> (32, 70, 70)

            nn.Conv2d(32, 32, kernel_size=5),     # -> (32, 66, 66)
            nn.ReLU(),
            nn.MaxPool2d(2),                      # -> (32, 33, 33)

            nn.Conv2d(32, 64, kernel_size=4),     # -> (64, 30, 30)
            nn.ReLU(),
            nn.MaxPool2d(2),                      # -> (64, 15, 15)

            nn.Conv2d(64, 64, kernel_size=4),     # -> (64, 12, 12)
            nn.ReLU(),
            nn.MaxPool2d(2),                      # -> (64, 6, 6)

            nn.Conv2d(64, 128, kernel_size=3),    # -> (128, 4, 4)
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2))          # -> (128, 2, 2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                         # 128*2*2 = 512
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)                     # for multi-class classification
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

