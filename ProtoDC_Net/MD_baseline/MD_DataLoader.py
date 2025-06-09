import os, random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.3665, 0.3665, 0.3665], std=[0.1911, 0.1911, 0.1911])
])


class MultiViewDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data = []
        self.classes = ['BrightLine', 'Deformation', 'Dent', 'Scratch', 'Normal']
        self.views = ['Down', 'Upper', 'Left', 'Right']

        print(f"MultiViewDataset({os.path.basename(self.root_dir)}): classes: {self.classes}, View: {self.views}.")

        for label, class_name in enumerate(self.classes):
            single_view_dir = os.path.join(self.root_dir, class_name, self.views[0])
            file_names = os.listdir(single_view_dir)
            for file_name in file_names:
                views_paths = []
                for view in self.views:
                    modified_file_name = file_name.replace("Down", view)
                    file_path = os.path.join(self.root_dir, class_name, view, modified_file_name)
                    views_paths.append(file_path)
                if len(views_paths) == len(self.views):
                    self.data.append((views_paths, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        views_paths, label = self.data[idx]
        images_tensor = torch.stack([
            transform(Image.open(view_path).convert("RGB"))
            for view_path in views_paths
        ])  # [4, C, H, W]

        return images_tensor, label


if __name__ == '__main__':
    train_dir = '/home/johs/Multi-light_source_USB-Connection/patch_split/train'
    train_dataset = MultiViewDataset(root_dir=train_dir)
