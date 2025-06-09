import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
CUDA = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from SD_baseline.SD4_DataLoader import SV4Dataset
from AnomalyLoss import cosine_compactness_loss
from AnomalyDetection import build_prototype, get_val_threshold, evaluate_anomaly_binary
from util import *
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

epochs = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataSet, dataLoader
# num_cores = os.cpu_count()
# num_workers = max(1, num_cores // 2)  # 코어 수의 50%로 설정

aug1_list = ["rotate", "rotate", "rotate", "noise"]
aug2_list = ["crop", "blur", "cutout", "rotate"]

cls_name = "zipper"
train_dir = f'/home/johs/Multi-light_source_USB-Connection/MVTec_AD/{cls_name}/train'
valid_dir = f'/home/johs/Multi-light_source_USB-Connection/MVTec_AD/{cls_name}/valid'
test_dir = f'/home/johs/Multi-light_source_USB-Connection/MVTec_AD/{cls_name}/test'
train_dataset = SV4Dataset(root_dir=train_dir, aug1_list=aug1_list, aug2_list=aug2_list)
valid_dataset = SV4Dataset(root_dir=valid_dir)
test_dataset = SV4Dataset(root_dir=test_dir, mode='test')
train_loader = DataLoader(train_dataset, batch_size=2 ** 5, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=2 ** 5, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2 ** 5, shuffle=False)

weight_path = f'/home/johs/Multi-light_source_USB-Connection/weights/Anomaly/Anomaly_{cls_name}.pth'

model = cosine_compactness_loss().to(device)
model.load_state_dict(torch.load(weight_path))

print(f"\nTest model {model.__class__.__name__}, GPU: {CUDA}, Weight path: {os.path.basename(weight_path)}\n")

# 🧪 Validation prototype accuracy
prototypes = build_prototype(model.EmbeddingInput, train_loader, device)
threshold = get_val_threshold(model.EmbeddingInput, valid_loader, prototypes, device)

for margin in [1.0000, 1.0002, 1.0004, 1.0006, 1.0008]:
    threshold, acc, normal_acc, abnormal_acc, f1 = evaluate_anomaly_binary(
        model.EmbeddingInput, test_loader, prototypes, threshold, device, margin=margin
    )
    print(f"margin: {margin:.4f} | threshold: {threshold:.4f} | acc: {acc:.2f}, normal_acc: {normal_acc:.2f}, abnormal_acc: {abnormal_acc:.2f}, f1: {f1:.2f}")
