import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
CUDA = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from SD4_DataLoader import SV4Dataset
from ProtoDC_Net.ProtoDC_loss import ProtoDC_loss
from util import *
from ProtoDC_Net.Prototype_Classifier import build_prototypes, evaluate_prototype
import torch.optim as optim
import time
import random
import numpy as np

SEED = 43
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

epochs = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataSet, dataLoader
num_cores = os.cpu_count()
num_workers = max(1, num_cores // 2)  # 코어 수의 50%로 설정

aug1_list = ["rotate", "rotate", "rotate", "noise"]
aug2_list = ["crop", "blur", "cutout", "rotate"]

train_dir = '/home/johs/Multi-light_source_USB-Connection/DAGM2007/train'
valid_dir = '/home/johs/Multi-light_source_USB-Connection/DAGM2007/valid'
train_dataset = SV4Dataset(root_dir=train_dir, aug1_list=aug1_list, aug2_list=aug2_list)
valid_dataset = SV4Dataset(root_dir=valid_dir)
train_loader = DataLoader(train_dataset, batch_size=2 ** 5, shuffle=True, num_workers=num_workers)
valid_loader = DataLoader(valid_dataset, batch_size=2 ** 5, num_workers=num_workers)

# Set up model and tools
model = ProtoDC_loss().to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0002)

weight_path = f'/home/johs/Multi-light_source_USB-Connection/weights/seed43_DAGM_ProtoDC_loss.pth'

print(f"\nTraining model {model.__class__.__name__}, GPU: {CUDA}, "
      f"Optimizer: {optimizer.__class__.__name__}, Weight path: {os.path.basename(weight_path)}\n")

total_time = time.time()
best_accuracy = 0.0
for epoch in range(1, epochs + 1):

    # Train model
    model.train()
    epoch_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        loss = model(images, labels)

        # Backward pass, Update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # scheduler.step()
    print(f"Epoch {epoch}/{epochs} | Loss: {epoch_loss: .4f}")

    # 🧪 Validation prototype accuracy
    prototypes = build_prototypes(model.EmbeddingInput, train_loader, device)
    valid_matrix, val_acc = evaluate_prototype(model.EmbeddingInput, valid_loader, prototypes, device)

    if val_acc > best_accuracy:
        best_accuracy = val_acc
        torch.save(model.state_dict(), weight_path)
        print(f"Saved weight_path: {weight_path}, Valid acc: {val_acc * 100: .2f}%\n")

    if epoch % 10 == 0:
        print(
            f"Loss: {epoch_loss: .4f}, Weight: {weight_path}, Accumulated time: {(time.time() - total_time) / 3600: .2f} hours. Valid matrix: \n{valid_matrix}")
        calculate_accuracies(valid_matrix)

    if epoch % 100 == 0:
        print(f'Epoch: {epoch}/{epochs}. Middle training Time: {(time.time() - total_time) / 3600: .2f} hours')
        # torch.save(model.state_dict(), weight_path2)
        # print(f"Saved weight_path: {weight_path}, Valid acc: {val_acc * 100: .2f}%\n")

print(f'\nTraining end. Total epoch: {epochs}, Total training Time: {((time.time() - total_time) / 3600): .2f} hours\n')
