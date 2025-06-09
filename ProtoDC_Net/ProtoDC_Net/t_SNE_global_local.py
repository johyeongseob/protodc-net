"""
Ensemble baseline model using independent processing of multiple illumination inputs.

Inspired by:
Guizhong Fu, Shukai Jia, Wenbin Zhu, Jiangxin Yang, Yanlong Cao, Michael Ying Yang, and Yanpeng Cao.  
"Fusion of multi-light source illuminated images for effective defect inspection on highly reflective surfaces."  
Mechanical Systems and Signal Processing, 175:109109, 2022.

Unlike the original fusion-based architecture, this model treats each illumination-specific image independently.  
Each view is processed by a separate SqueezeNet branch, and the final prediction is obtained by averaging  
the logits across all views. No feature-level fusion is performed.
"""


import torch
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import DataLoader
from MD_baseline.MD_DataLoader import MultiViewDataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from ProtoDC_loss import ProtoDC_loss
import numpy as np
import matplotlib.lines as mlines


class EmbeddingVisualizer:
    def __init__(self, model, device):
        """
        model: An embedding model such as SupCLIPLoss
        device: torch.device("cuda" or "cpu")
        """
        self.model = model
        self.device = device

    def extract_embeddings(self, dataloader):
        self.model.eval()
        all_local_embs, all_global_embs, all_labels = [], [], []

        with torch.no_grad():
            for images, labels in dataloader:  # images: [B, 4, 3, H, W]
                images, labels = images.to(self.device), labels.to(self.device)
                local_embs, global_embs = self.model.EmbeddingInput(images)
                all_local_embs.append(local_embs.cpu())
                all_global_embs.append(global_embs.cpu())
                all_labels.append(labels.cpu())

        local_embs = torch.cat(all_local_embs)    # [N, C]
        global_embs = torch.cat(all_global_embs)  # [N, C]
        labels = torch.cat(all_labels)

        return local_embs.numpy(), global_embs.numpy(), labels.numpy()

    def visualize_tsne(self, dataloader, save_path=None):
        local_embs, global_embs, labels = self.extract_embeddings(dataloader)

        num_classes = len(np.unique(labels))
        local_protos, global_protos, proto_labels, proto_types = [], [], [], []

        for cls_idx in range(num_classes):
            # Local prototype
            local_mask = labels == cls_idx
            local_proto = local_embs[local_mask].mean(axis=0, keepdims=True)
            local_protos.append(local_proto)
            proto_labels.append(cls_idx)
            proto_types.append('local_proto')

            # Global prototype
            global_mask = labels == cls_idx
            global_proto = global_embs[global_mask].mean(axis=0, keepdims=True)
            global_protos.append(global_proto)
            proto_labels.append(cls_idx)
            proto_types.append('global_proto')

        # Concatenate all embeddings
        embeddings = np.concatenate([local_embs, global_embs] + local_protos + global_protos, axis=0)
        label_list = np.concatenate([labels, labels, proto_labels])  # total: 2N + 10
        type_flag = np.array(['local'] * len(labels) + ['global'] * len(labels) +
                             ['local_proto'] * num_classes + ['global_proto'] * num_classes)

        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        reduced = tsne.fit_transform(embeddings)

        # Define high-contrast colors for visibility
        high_contrast_colors = {
            'BrightLine': '#000000',   # black
            'Deformation': '#0072B2',  # blue
            'Dent': '#D55E00',         # reddish orange
            'Scratch': '#009E73',      # green
            'Normal': '#E69F00'        # orange
        }

        # Visualization
        plt.figure(figsize=(10, 8))
        cmap = plt.get_cmap('tab10')
        markers = {
            'local': 'o',
            'global': 'x',
            'global_proto': 'P'
        }

        for emb_type in ['local', 'global', 'global_proto']:
            for cls_idx, cls_name in enumerate(['BrightLine', 'Deformation', 'Dent', 'Scratch', 'Normal']):
                idxs = (type_flag == emb_type) & (label_list == cls_idx)

                # Do not show labels for global prototypes
                if emb_type == 'global_proto':
                    label_text = None
                else:
                    label_text = f"{emb_type} - {cls_name}"

                plt.scatter(
                    reduced[idxs, 0], reduced[idxs, 1],
                    color='white' if 'proto' in emb_type else high_contrast_colors[cls_name],
                    marker=markers[emb_type],
                    alpha=1.0 if 'proto' in emb_type else 0.6,
                    edgecolors='k' if 'proto' in emb_type else 'none',
                    s=130 if 'proto' in emb_type else 30,
                    label=label_text
                )

        # Manually add prototype marker to legend
        proto_marker = mlines.Line2D(
            [], [],
            marker='P',
            color='white',
            markeredgecolor='black',
            markerfacecolor='white',
            linestyle='None',
            markersize=10,
            label='Prototype'
        )

        # Combine with existing legend
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(proto_marker)
        labels.append('Prototype')
        plt.legend(handles=handles, labels=labels, loc='lower right', fontsize=12)

        # Axis and grid

        plt.xlim(-80, 80)
        plt.ylim(-80, 80)
        plt.xlabel("Component 1", fontsize=18)
        plt.ylabel("Component 2", fontsize=18)
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Saved to {save_path}")
        else:
            plt.show()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ProtoDC_loss().to(device)

    weight_path = f'weights/ProtoDC_main.pth'
    model.load_state_dict(torch.load(weight_path))

    train_dir = 'USB_MD/train'
    train_dataset = MultiViewDataset(root_dir=train_dir)
    train_loader = DataLoader(train_dataset, batch_size=2 ** 5, shuffle=False)

    # Run t-SNE visualization
    visualizer = EmbeddingVisualizer(model, device)
    visualizer.visualize_tsne(train_loader, save_path="tsne_after.png")
