import json
import os
import random

from PIL import Image, ImageOps
from PIL.Image import Resampling
from torch.utils.data import Dataset


class DatasetTrain(Dataset):
    def __init__(self, root, num_of_users=100, transforms_sketch=None, transforms_image=None):
        if num_of_users > 100:
            raise ValueError('num_of_users > 100')

        self.root = root

        with open(os.path.join(self.root, 'val_normal.txt'), 'r') as f:
            lines = f.readlines()
            val_ids = list(map(int, lines))

        self.files = []
        for i in range(1, num_of_users + 1):
            file_names = os.listdir(os.path.join(root, "images", str(i)))
            file_names = [file.split('.')[0] for file in file_names]
            for file_name in file_names:
                if int(file_name) not in val_ids:
                    self.files.append(os.path.join(str(i), file_name))

        self.transforms_sketch = transforms_sketch
        self.transforms_image = transforms_image

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.root, "raster_sketches", self.files[idx] + ".jpg")
        image_path = os.path.join(self.root, "images", self.files[idx] + ".jpg")

        negative_idx = random.randint(0, len(self.files) - 1)
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.files) - 1)

        negative_path = os.path.join(self.root, "images", self.files[negative_idx] + ".jpg")

        sketch = Image.open(sketch_path)
        # sketch = ImageOps.pad(sketch, (224, 224))

        image = Image.open(image_path)
        # image = ImageOps.pad(image, (224, 224))

        negative = Image.open(negative_path)
        # negative = ImageOps.pad(negative, (224, 224))

        if self.transforms_sketch:
            sketch = self.transforms_sketch(sketch)

        if self.transforms_image:
            image = self.transforms_image(image)
            negative = self.transforms_image(negative)

        return sketch, image, negative


class DatasetTest(Dataset):
    def __init__(self, root, num_of_users=100, transforms=None):
        if num_of_users > 100:
            raise ValueError('num_of_users > 100')

        self.root = root

        with open(os.path.join(self.root, "..", 'val_normal.txt'), 'r') as f:
            lines = f.readlines()
            val_ids = list(map(int, lines))

        self.files = []
        for i in range(1, num_of_users + 1):
            file_names = os.listdir(os.path.join(root, str(i)))
            file_names = [file.split('.')[0] for file in file_names]
            for file_name in file_names:
                if int(file_name) in val_ids:
                    self.files.append(os.path.join(str(i), file_name))

        self.files = sorted(self.files)

        self.transforms = transforms

    def get_file_names(self):
        return self.files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        img_path = os.path.join(self.root, self.files[idx] + ".jpg")
        img = Image.open(img_path)
        # img = ImageOps.pad(img, (224, 224))

        if self.transforms is not None:
            img = self.transforms(img)

        return img
