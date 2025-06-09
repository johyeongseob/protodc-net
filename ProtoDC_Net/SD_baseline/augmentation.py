import os
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import random
import numpy as np
import torch


# 데이터 증강 구성

class RandomCrop:
    def __init__(self, size=(224, 224), scale=(0.08, 1.0), ratio=(3/4, 4/3)):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5)
        ])

    def __call__(self, img):
        return self.transform(img)


class ColorDistortion:
    def __init__(self, s=1.0):
        """
        s: distortion strength (기본 1.0, SimCLR 스타일)
        """
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)

        self.transform = transforms.Compose([
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ])

    def __call__(self, x):
        return self.transform(x)


class RandomRotation:
    def __call__(self, x):
        angle = random.choice([0, 90, 180, 270])
        return F.rotate(x, angle)


class Cutout:
    def __init__(self):
        self.size_ratio = 0.5

    def __call__(self, img):
        np_img = np.array(img).copy()
        h, w = np_img.shape[:2]

        size = int(self.size_ratio * w)
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)

        x1 = max(0, x - size // 2)
        y1 = max(0, y - size // 2)
        x2 = min(w, x + size // 2)
        y2 = min(h, y + size // 2)

        np_img[y1:y2, x1:x2] = (0, 0, 0)
        return Image.fromarray(np_img)

class GaussianBlur:
    def __init__(self):
        self.kernel_ratio = 0.1
        self.sigma_min = 0.1
        self.sigma_max = 2.0

    def __call__(self, img):
        np_img = np.array(img)
        h, w = np_img.shape[:2]

        k = max(3, int(self.kernel_ratio * w))
        if k % 2 == 0:
            k += 1

        sigma = random.uniform(self.sigma_min, self.sigma_max)
        blurred = cv2.GaussianBlur(np_img, ksize=(k, k), sigmaX=sigma)
        return Image.fromarray(blurred)


class GaussianNoise:
    def __init__(self):
        """
        std_min, std_max: 노이즈 표준편차 범위
        """
        self.to_tensor = transforms.ToTensor()
        self.to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img_tensor = self.to_tensor(img)
        std = random.uniform(0.0, 0.1)

        noisy = img_tensor + torch.randn_like(img_tensor) * std
        noisy = torch.clamp(noisy, 0.0, 1.0)

        return self.to_pil(noisy)


class SobelFilter:
    def __call__(self, img):
        """
        입력: PIL.Image (RGB)
        출력: PIL.Image (Sobel edge를 RGB로 복원)
        """
        img_gray = np.array(img.convert("L"))  # (H, W) 흑백

        sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

        sobel_edge = np.hypot(sobel_x, sobel_y)
        sobel_edge = sobel_edge / (sobel_edge.max() + 1e-8) * 255.0
        sobel_edge = sobel_edge.astype(np.uint8)

        sobel_rgb = np.stack([sobel_edge] * 3, axis=-1)  # (H, W, 3)
        return Image.fromarray(sobel_rgb)


class HistogramEqualization:
    def __call__(self, img):
        """
        입력: PIL.Image (RGB 또는 L)
        출력: 히스토그램 평활화된 PIL.Image (RGB)
        """
        img = img.convert("L")  # 그레이스케일로 변환
        img_np = np.array(img)

        equalized_np = cv2.equalizeHist(img_np)

        # 3채널 RGB로 변환 (ToTensor와 호환되도록)
        equalized_rgb = np.stack([equalized_np]*3, axis=-1)
        return Image.fromarray(equalized_rgb)
