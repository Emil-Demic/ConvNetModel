import os
import random

import cv2
import numpy
from PIL import Image
from torch.utils.data import Dataset


class DatasetTrain(Dataset):
    def __init__(self, root, transforms_sketch=None, transforms_image=None):
        self.root = root
        self.sketches = os.listdir(os.path.join(self.root, "sketch", "Image"))
        self.images = os.listdir(os.path.join(self.root, "image", "Image"))
        self.transforms_sketch = transforms_sketch
        self.transforms_image = transforms_image
        random.seed(42)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.root, "sketch", "Image", self.sketches[idx])
        image_path = os.path.join(self.root, "image", "Image", self.images[idx])

        negative_idx = random.randint(0, len(self.sketches) - 1)
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.sketches) - 1)

        negative_path = os.path.join(self.root, "image", "Image", self.images[negative_idx])

        sketch = Image.open(sketch_path)
        image = Image.open(image_path).convert('RGB')
        negative = Image.open(negative_path).convert('RGB')

        if self.transforms_sketch:
            sketch = self.transforms_sketch(sketch)

        if self.transforms_image:
            open_cv_image = numpy.array(image)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            edge_map = cv2.Canny(open_cv_image, 150, 300)
            image = Image.fromarray(edge_map)
            image = self.transforms_image(image)

            open_cv_image = numpy.array(negative)
            open_cv_image = open_cv_image[:, :, ::-1].copy()
            edge_map = cv2.Canny(open_cv_image, 150, 300)
            negative = Image.fromarray(edge_map)
            negative = self.transforms_image(negative)

        return sketch, image, negative


class DatasetTest(Dataset):
    def __init__(self, img_dir, transforms=None, sketch=True):
        self.img_dir = img_dir
        self.images = os.listdir(img_dir)
        self.transforms = transforms
        self.sketch = sketch

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        img = Image.open(img_path).convert('RGB')
        if self.transforms is not None:
            if not self.sketch:
                open_cv_image = numpy.array(img)
                open_cv_image = open_cv_image[:, :, ::-1].copy()
                edge_map = cv2.Canny(open_cv_image, 150, 300)
                img = Image.fromarray(edge_map)
            img = self.transforms(img)

        return img


