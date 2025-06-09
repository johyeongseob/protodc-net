import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
CUDA = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from SD4_DataLoader import SV4Dataset
from ProtoDC_Net.ProtoDC_loss import ProtoDC_loss
from util import *
from ProtoDC_Net.Prototype_Classifier import build_prototypes, evaluate_prototype
import random
import numpy as np

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

epochs = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataSet, dataLoader
# num_cores = os.cpu_count()
# num_workers = max(1, num_cores // 2)  # 코어 수의 50%로 설정

aug1_list = ["rotate", "rotate", "rotate", "noise"]
aug2_list = ["crop", "blur", "cutout", "rotate"]

train_dir = '/home/johs/Multi-light_source_USB-Connection/USB_SD/train'
test_dir = '/home/johs/Multi-light_source_USB-Connection/USB_SD/valid'
train_dataset = SV4Dataset(root_dir=train_dir, aug1_list=aug1_list, aug2_list=aug2_list)
test_dataset = SV4Dataset(root_dir=test_dir)
train_loader = DataLoader(train_dataset, batch_size=2 ** 5, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=2 ** 5, shuffle=False)

weight_path = f'/home/johs/Multi-light_source_USB-Connection/weights/seed42_SD_ProtoDC_loss.pth'

model = ProtoDC_loss().to(device)
model.load_state_dict(torch.load(weight_path))

print(f"\nTest model {model.__class__.__name__}, GPU: {CUDA}, Weight path: {os.path.basename(weight_path)}\n")

# 🧪 Validation prototype accuracy
prototypes = build_prototypes(model.EmbeddingInput, train_loader, device)
test_matrix, test_acc = evaluate_prototype(model.EmbeddingInput, test_loader, prototypes, device)

print(f"Test matrix: \n{test_matrix}")
calculate_accuracies(test_matrix)
