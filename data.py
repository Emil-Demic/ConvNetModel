import os
import random

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
        image = Image.open(image_path)
        negative = Image.open(negative_path)

        if self.transforms_sketch:
            sketch = self.transforms_sketch(sketch)

        if self.transforms_image:
            image = self.transforms_image(image)
            negative = self.transforms_image(negative)

        return sketch, image, negative


class DatasetTest(Dataset):
    def __init__(self, img_dir, transforms=None):
        self.img_dir = img_dir
        self.images = os.listdir(img_dir)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        img = Image.open(img_path)
        if self.transforms is not None:
            img = self.transforms(img)

        return img


