import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
CUDA = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from MD_baseline.MD_DataLoader import MultiViewDataset
from Prototype_Classifier import build_prototypes, evaluate_prototype
from ProtoDC_loss import ProtoDC_loss
from Ablation_global import ProtoDC_global
from Ablation_local import ProtoDC_local
from Ablation_nomedian import ProtoDC_nomedian
from util import *
from sklearn.metrics import confusion_matrix, accuracy_score
import torch.optim as optim
import random
import numpy as np

SEED = 44
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataSet, dataLoader
num_cores = os.cpu_count()
num_workers = max(1, num_cores // 2)  # 코어 수의 50%로 설정

train_dir = '/home/johs/Multi-light_source_USB-Connection/USB_MD/train'
test_dir = '/home/johs/Multi-light_source_USB-Connection/USB_MD/test'
train_dataset = MultiViewDataset(root_dir=train_dir)
test_dataset = MultiViewDataset(root_dir=test_dir)
train_loader = DataLoader(train_dataset, batch_size=2 ** 5, shuffle=True, num_workers=num_workers)
test_loader = DataLoader(test_dataset, batch_size=2 ** 5, num_workers=num_workers)

weight_path = f'weights/seed44_ProtoDC_loss.pth'

model = ProtoDC_nomedian().to(device)
model.load_state_dict(torch.load(weight_path))

# 🧪 Test prototype accuracy
prototypes = build_prototypes(model.EmbeddingInput, train_loader, device)
test_matrix, test_acc = evaluate_prototype(model.EmbeddingInput, test_loader, prototypes, device)

# # Embedding Visualization
# visualizer = EmbeddingVisualizer(model, device)
# visualizer.visualize_tsne(train_loader, save_path=f"4SupCLIP/tsne_embedding_train.png")

print(f"weight_path: {weight_path}")
print(f"Test matrix: \n{test_matrix}")
calculate_accuracies(test_matrix)
