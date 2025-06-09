import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
CUDA = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from SD_baseline.SD4_DataLoader import SV4Dataset
from util import *
from AnomalyDetection import build_prototype, get_val_threshold
from AnomalyLoss import cosine_compactness_loss
import torch.optim as optim
import time
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
num_cores = os.cpu_count()
num_workers = max(1, num_cores // 2)  # 코어 수의 50%로 설정

aug1_list = ["rotate", "rotate", "rotate", "noise"]
aug2_list = ["crop", "blur", "cutout", "rotate"]

train_dir = '/home/johs/Multi-light_source_USB-Connection/MVTec_AD/zipper/train'
valid_dir = '/home/johs/Multi-light_source_USB-Connection/MVTec_AD/zipper/valid'
train_dataset = SV4Dataset(root_dir=train_dir, aug1_list=aug1_list, aug2_list=aug2_list)
valid_dataset = SV4Dataset(root_dir=valid_dir)
train_loader = DataLoader(train_dataset, batch_size=2 ** 5, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=2 ** 5, num_workers=num_workers)

# Set up model and tools
model = cosine_compactness_loss().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0002)

weight_path = f'/home/johs/Multi-light_source_USB-Connection/weights/Anomaly/Anomaly_zipper.pth'

print(f"\nTraining model {model.__class__.__name__}, GPU: {CUDA}, "
      f"Optimizer: {optimizer.__class__.__name__}, Weight path: {os.path.basename(weight_path)}\n")

total_time = time.time()
best_threshold = -1e9
for epoch in range(1, epochs + 1):

    # Train model
    model.train()
    epoch_loss = 0.0

    for images, _ in train_loader:
        images = images.to(device)

        loss = model(images)

        # Backward pass, Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # scheduler.step()
    print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss: .4f}")

    # 🧪 Validation prototype accuracy
    prototypes = build_prototype(model.EmbeddingInput, train_loader, device)
    threshold = get_val_threshold(model.EmbeddingInput, valid_loader, prototypes, device)

    if threshold > best_threshold:
        best_threshold = threshold
        torch.save(model.state_dict(), weight_path)
        print(f"Saved weight_path: {weight_path}, best_threshold: {best_threshold: .4f}\n")

    if epoch % 10 == 0:
        print(f"Loss: {epoch_loss: .4f}, best_threshold: {best_threshold: .4f}, Weight: {weight_path}")

print(f'\nTraining end. Total epoch: {epochs}, Total training Time: {((time.time() - total_time) / 3600): .2f} hours\n')
