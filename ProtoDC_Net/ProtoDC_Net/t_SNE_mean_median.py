import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
from torch.utils.data import DataLoader
from MD_baseline.MD_DataLoader import MultiViewDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from ProtoDC_loss import ProtoDC_loss
from Ablation_nomedian import ProtoDC_nomedian
import numpy as np


class EmbeddingVisualizer:
    def __init__(self, model1, model2, device):
        """
        model: ProtoDC와 같은 embedding 모델
        device: torch.device("cuda" or "cpu")
        """
        self.model1 = model1
        self.model2 = model2
        self.device = device

    def extract_embeddings(self, dataloader):
        self.model1.eval()
        self.model2.eval()
        local_mean_embs, local_median_embs, all_labels = [], [], []

        with torch.no_grad():
            for images, labels in dataloader:  # images: [B, 4, 3, H, W]
                images, labels = images.to(self.device), labels.to(self.device)
                local_mean, _ = self.model1.EmbeddingInput(images)
                local_median, _ = self.model2.EmbeddingInput(images)
                local_mean_embs.append(local_mean.cpu())
                local_median_embs.append(local_median.cpu())
                all_labels.append(labels.cpu())

        mean_embs = torch.cat(local_mean_embs)  # [N, C]
        median_embs = torch.cat(local_median_embs)  # [N, C]
        labels = torch.cat(all_labels)

        return mean_embs.numpy(), median_embs.numpy(), labels.numpy()

    def visualize_tsne(self, dataloader, save_path=None):
        mean_embs, median_embs, labels = self.extract_embeddings(dataloader)

        num_classes = len(np.unique(labels))

        # concat
        embeddings = np.concatenate([mean_embs, median_embs], axis=0)
        label_list = np.concatenate([labels, labels])  # total: 2N
        type_flag = np.array(['mean'] * len(labels) + ['median'] * len(labels))

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(embeddings)

        # 시각화
        plt.figure(figsize=(10, 8))
        cmap = plt.get_cmap('tab10')  # 최대 10개 클래스 색상 지원
        markers = {
            'mean': 'o',
            'median': 'x'
        }

        class_names = ['BrightLine', 'Deformation', 'Dent', 'Scratch', 'Normal']

        for emb_type in ['mean', 'median']:
            for cls_idx, cls_name in enumerate(class_names):
                idxs = (type_flag == emb_type) & (label_list == cls_idx)

                plt.scatter(
                    reduced[idxs, 0],  # x좌표
                    reduced[idxs, 1],  # y좌표
                    marker=markers[emb_type],
                    color=cmap(cls_idx),
                    label=f"{cls_name} ({emb_type})",
                    alpha=0.7,
                    edgecolor='k',
                    linewidths=0.5
                )

        plt.title("t-SNE: Local(mean/median) Embeddings")
        plt.xlabel("Component 1")
        plt.ylabel("Component 2")
        plt.legend(loc='best', fontsize=9)
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Saved to {save_path}")
        else:
            plt.show()



if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model1 = ProtoDC_nomedian().to(device)
    model2 = ProtoDC_loss().to(device)

    weight_path1 = f'weights/ProtoDC_nomedian.pth'
    weight_path2 = f'weights/ProtoDC_main.pth'
    model1.load_state_dict(torch.load(weight_path1))
    model2.load_state_dict(torch.load(weight_path2))

    train_dir = 'USB_MD/train'
    train_dataset = MultiViewDataset(root_dir=train_dir)
    train_loader = DataLoader(train_dataset, batch_size=2 ** 5, shuffle=False)

    # Visualize
    visualizer = EmbeddingVisualizer(model1, model2, device)
    visualizer.visualize_tsne(train_loader, save_path="tsne_local.png")
