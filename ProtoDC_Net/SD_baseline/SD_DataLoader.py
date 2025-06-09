import os
from augmentation import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as f
from PIL import Image


def get_transform(aug1="color", aug2="crop"):
    AUGMENT_MAP = {
        "none": lambda: None,
        "crop": RandomCrop,
        "color": ColorDistortion,
        "rotate": RandomRotation,
        "cutout": Cutout,
        "blur": GaussianBlur,
        "noise": GaussianNoise,
        "sobel": SobelFilter,
        # "histogram": HistogramEqualization
    }

    transform_list = [transforms.Resize((224, 224))]
    # transform_list = []

    for aug in [aug1, aug2]:
        if aug not in AUGMENT_MAP:
            raise ValueError(f"Unknown augmentation type: {aug}")
        aug_transform = AUGMENT_MAP[aug]()
        print(f"aug_transform: {aug_transform}")
        if aug_transform is not None:
            transform_list.append(aug_transform)

    transform_list += [
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5642] * 3, std=[0.1709] * 3)  # USB-SD
        transforms.Normalize(mean=[0.4602] * 3, std=[0.1832] * 3)  # DAGM2007
    ]

    return transforms.Compose(transform_list)


class SVDataset(Dataset):
    def __init__(self, root_dir=None, aug1="none", aug2="none"):
        self.root_dir = root_dir
        self.transform = get_transform(aug1, aug2)
        self.data = []
        # self.classes = ['BrightLine', 'Deformation', 'Dent', 'Scratch', 'Spot', 'Squalidity', 'Normal']  # USB-SD
        self.classes = ['Class1', 'Class1_def', 'Class2', 'Class2_def', 'Class3', 'Class3_def',
                        'Class4', 'Class4_def', 'Class5', 'Class5_def', 'Class6', 'Class6_def']

        print(f"SingleViewDataset. classes: {self.classes}.")

        # 이미지 경로와 레이블을 한 번에 생성
        for label, class_name in enumerate(self.classes):
            dir_path = os.path.join(self.root_dir, class_name)
            for file_name in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file_name)
                self.data.append((file_path, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)  # [C, H, W]
        fname = os.path.basename(img_path)
        return image, label


if __name__ == '__main__':
    test_dir = 'USB_SD/test'
    test_dataset = SVDataset(root_dir=test_dir, aug1='crop', aug2="color")
    test_loader = DataLoader(test_dataset, batch_size=2 ** 2, shuffle=False)

    print(f"Test dataset size: {len(test_dataset)}")

    for images, labels in test_loader:
        for i in range(len(images)):
            img = f.to_pil_image(images[i])  # tensor → PIL 이미지로 변환
            plt.subplot(1, len(images), i + 1)
            plt.imshow(img)
            plt.title(f"Label: {labels[i].item()}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()
        break
