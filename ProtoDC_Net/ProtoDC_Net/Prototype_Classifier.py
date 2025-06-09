import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
from util import *


def build_prototypes(embedding_model, data_loader, device):
    embedding_model.eval()
    class_to_vectors = {}

    with torch.no_grad():
        for views, labels in data_loader:
            views, labels = views.to(device), labels.to(device)
            _, global_embeddings = embedding_model(views)
            # global_embeddings = embedding_model(views)
            global_embeddings = F.normalize(global_embeddings, dim=1)

            # Group each embedding by its class label for prototype construction
            for emb, label in zip(global_embeddings, labels):
                label = label.item()
                class_to_vectors.setdefault(label, []).append(emb.unsqueeze(0))

    prototypes = {}
    for cls, vectors in class_to_vectors.items():  # class 개수만큼 반복
        proto = torch.cat(vectors, dim=0).mean(dim=0, keepdim=True)
        prototypes[cls] = F.normalize(proto, dim=1)

    return prototypes


def evaluate_prototype(embedding_model, data_loader, prototypes, device):
    embedding_model.eval()
    y_true, y_pred, wrong_samples = [], [], []

    with torch.no_grad():
        proto_matrix = torch.cat([prototypes[c] for c in sorted(prototypes)], dim=0)  # [K, C]
        for views, labels in data_loader:
            views, labels = views.to(device), labels.to(device)

            _, global_embeddings = embedding_model(views)
            # global_embeddings = embedding_model(views)
            global_embeddings = F.normalize(global_embeddings, dim=1)  # [B, C]
            sim = torch.matmul(global_embeddings, proto_matrix.T)  # [B, K]
            pred = sim.argmax(dim=1)  # 행 별로 가장 높은 값으로 예측

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())


    cm = confusion_matrix(y_true, y_pred)
    acc = np.mean(np.array(y_true) == np.array(y_pred))

    return cm, acc
