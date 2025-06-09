import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '')))
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def build_prototype(embedding_model, data_loader, device):
    embedding_model.eval()
    vectors = []

    with torch.no_grad():
        for views, _ in data_loader:
            views = views.to(device)
            _, global_embeddings = embedding_model(views)
            # global_embeddings = embedding_model(views)
            global_embeddings = F.normalize(global_embeddings, dim=1)

            vectors.append(global_embeddings)

    all_vectors = torch.cat(vectors, dim=0)  # [N, C]

    proto = all_vectors.mean(dim=0, keepdim=True)
    prototype = F.normalize(proto, dim=1)

    return prototype  # [1, C]


# Threshold

def get_val_threshold(model, val_loader, prototype, device):
    model.eval()
    similarities = []

    with torch.no_grad():
        for views, _ in val_loader:
            views = views.to(device)
            _, global_emb = model(views)
            global_emb = F.normalize(global_emb, dim=1)
            sim = F.cosine_similarity(global_emb, prototype, dim=1)  # [B]
            similarities.append(sim)

    similarities = torch.cat(similarities)  # [N]
    threshold = similarities.min().item()   # 가장 낮은 유사도 값
    return threshold


def evaluate_anomaly_binary(model, test_loader, prototype, threshold, device, margin=1):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for views, labels in test_loader:
            views = views.to(device)
            _, global_emb = model(views)
            global_emb = F.normalize(global_emb, dim=1)
            sim = F.cosine_similarity(global_emb, prototype, dim=1)  # [B]
            adjusted_threshold = threshold * margin

            # cosine similarity < threshold → anomaly (label 1)
            preds = (sim < adjusted_threshold).long()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # 정상(0)과 비정상(1) 각각의 accuracy 계산
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)

    normal_mask = (y_true == 0)
    abnormal_mask = (y_true == 1)

    normal_acc = (y_pred[normal_mask] == 0).float().mean().item() * 100 if normal_mask.sum() > 0 else 0.0
    abnormal_acc = (y_pred[abnormal_mask] == 1).float().mean().item() * 100 if abnormal_mask.sum() > 0 else 0.0

    return adjusted_threshold, acc * 100, normal_acc, abnormal_acc, f1 * 100


# ____________________________________________________________________________________ #

# AUROC

# def avg_val_similarity(model, val_loader, prototype, device):
#     model.eval()
#     sims = []
#     with torch.no_grad():
#         for views, _ in val_loader:
#             views = views.to(device)
#             _, global_emb = model(views)
#             global_emb = F.normalize(global_emb, dim=1)
#             prototype = F.normalize(prototype, dim=1)
#             sim = F.cosine_similarity(global_emb, prototype, dim=1)
#             sims.append(sim)
#     sims = torch.cat(sims)
#     return sims.mean().item()


# def evaluate_anomaly_auroc(embedding_model, test_loader, prototype, device):
#     embedding_model.eval()
#     y_true, scores = [], []

#     with torch.no_grad():
#         for views, labels in test_loader:
#             views, labels = views.to(device), labels.to(device)
#             _, global_test = embedding_model(views)
#             global_test = F.normalize(global_test, dim=1)  # [B, C]
#             sim = torch.matmul(global_test, prototype.T)  # [B]
#             sim = sim.view(-1)  # [B]
#             anomaly_score = 1 - sim  # 유사도 낮을수록 이상

#             y_true.extend(labels.cpu().numpy())
#             scores.extend(anomaly_score.cpu().numpy())

#     auc = roc_auc_score(y_true, scores)

#     return auc*100