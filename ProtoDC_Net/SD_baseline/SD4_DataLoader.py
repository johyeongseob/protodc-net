import os
from SD_baseline.augmentation import *
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as f
from torchvision.transforms import Lambda
from PIL import Image


def get_transform(aug1="color", aug2="crop"):
    AUGMENT_MAP = {
        "none": lambda: Lambda(lambda x: x),
        "crop": RandomCrop,
        "color": ColorDistortion,
        "rotate": RandomRotation,
        "cutout": Cutout,
        "blur": GaussianBlur,
        "noise": GaussianNoise,
        "sobel": SobelFilter,
    }

    transform_list = [transforms.Resize((224, 224))]

    for aug in [aug1, aug2]:
        if aug not in AUGMENT_MAP:
            raise ValueError(f"Unknown augmentation type: {aug}")
        aug_transform = AUGMENT_MAP[aug]()
        if aug_transform is not None:
            transform_list.append(aug_transform)

    transform_list += [
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5642] * 3, std=[0.1709] * 3)  # USB-SD
        transforms.Normalize(mean=[0.4602] * 3, std=[0.1832] * 3)  # DAGM2007
        # transforms.Normalize(mean=[0.5353, 0.5316, 0.5480], std=[0.3668, 0.3667, 0.3499])  # MVTec_AD bottle
        # transforms.Normalize(mean=[0.3264, 0.4144, 0.4666], std=[0.1533, 0.2152, 0.2391])  # MVTec_AD cable
        # transforms.Normalize(mean=[0.6966, 0.6668, 0.6539], std=[0.2386, 0.2592, 0.2602])  # MVTec_AD capsule
        # transforms.Normalize(mean=[0.3628, 0.3498, 0.3561], std=[0.1449, 0.1397, 0.1275])  # MVTec_AD carpet
        # transforms.Normalize(mean=[0.4482, 0.4482, 0.4482], std=[0.1645, 0.1645, 0.1645])  # MVTec_AD grid
        # transforms.Normalize(mean=[0.2397, 0.1767, 0.1713], std=[0.1682, 0.0757, 0.0437])  # MVTec_AD hazelnut
        # transforms.Normalize(mean=[0.3914, 0.2623, 0.2196], std=[0.0609, 0.0407, 0.0305])  # MVTec_AD leather
        # transforms.Normalize(mean=[0.2147, 0.2369, 0.2386], std=[0.1605, 0.1826, 0.1537])  # MVTec_AD metal_nut
        # transforms.Normalize(mean=[0.3024, 0.3029, 0.3247], std=[0.3043, 0.3043, 0.2869])  # MVTec_AD pill
        # transforms.Normalize(mean=[0.7222, 0.7222, 0.7222], std=[0.1336, 0.1336, 0.1336])  # MVTec_AD screw
        # transforms.Normalize(mean=[0.4562, 0.4707, 0.4476], std=[0.1205, 0.1331, 0.1284])  # MVTec_AD tile
        # transforms.Normalize(mean=[0.2001, 0.1842, 0.1912], std=[0.2302, 0.2142, 0.1937])  # MVTec_AD toothbrush
        # transforms.Normalize(mean=[0.3869, 0.2765, 0.2416], std=[0.2065, 0.1433, 0.1170])  # MVTec_AD transistor
        # transforms.Normalize(mean=[0.6692, 0.4766, 0.3492], std=[0.0828, 0.0677, 0.0478])  # MVTec_AD wood
        # transforms.Normalize(mean=[0.4010, 0.4010, 0.4010], std=[0.3231, 0.3231, 0.3231])  # MVTec_AD zipper
    ]

    return transforms.Compose(transform_list)



class SV4Dataset(Dataset):
    def __init__(self, root_dir=None, aug1_list=None, aug2_list=None, mode=None):
        self.root_dir = root_dir

        if aug1_list is None:
            aug1_list = ["none"] * 4
        if aug2_list is None:
            aug2_list = ["none"] * 4
        assert len(aug1_list) == len(aug2_list) == 4, "aug1_list and aug2_list must have 4 elements"

        self.transforms = [get_transform(a1, a2) for a1, a2 in zip(aug1_list, aug2_list)]

        self.data = []
        # self.classes = ['good']
        # if mode == 'test':
        #     self.classes = ['good', 'anomaly']
        # self.classes = ['BrightLine', 'Deformation', 'Dent', 'Scratch', 'Spot', 'Squalidity', 'Normal']
        self.classes = ['Class1', 'Class1_def', 'Class2', 'Class2_def', 'Class3', 'Class3_def',
                        'Class4', 'Class4_def', 'Class5', 'Class5_def', 'Class6', 'Class6_def']
        print(f"SingleViewDataset. classes: {self.classes}.")

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
        image_tensor = [transform(image) for transform in self.transforms]
        return torch.stack(image_tensor), label  # [4, C, H, W]


if __name__ == '__main__':
    test_dir = 'USB_SD/test'
    test_dataset = SV4Dataset(root_dir=test_dir)
    test_loader = DataLoader(test_dataset, batch_size=2 ** 2, shuffle=False)

    print(f"Test dataset size: {len(test_dataset)}")

    for images, labels in test_loader:
        for i in range(len(images)):
            img = f.to_pil_image(images[i][3])  # tensor → PIL 이미지로 변환
            plt.subplot(1, len(images), i + 1)
            plt.imshow(img)
            plt.title(f"Label: {labels[i].item()}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()
        break
