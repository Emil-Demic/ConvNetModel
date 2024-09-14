import json
import os
import random

import numpy as np
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from config import args
from utils import drawPNG


class DatasetTrain(Dataset):
    def __init__(self, root, num_of_users=100, transforms_sketch=None, transforms_image=None):
        if num_of_users > 100:
            raise ValueError('num_of_users > 100')

        self.root = root

        with open(os.path.join(self.root, 'val_unseen_user.txt'), 'r') as f:
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

        self.rng = np.random.default_rng(seed=args.seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.root, "raw_data", self.files[idx] + ".json")
        image_path = os.path.join(self.root, "images", self.files[idx] + ".jpg")

        negative_idx = random.randint(0, len(self.files) - 1)
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.files) - 1)

        negative_path = os.path.join(self.root, "images", self.files[negative_idx] + ".jpg")

        selection = self.rng.choice([1, 2, 3, 4], p=[0.7, 0.1, 0.1, 0.1])
        amount = self.rng.random() % 0.1
        match selection:
            case 1:
                sketch = drawPNG(json.load(open(sketch_path)))
            case 2:
                sketch = drawPNG(json.load(open(sketch_path)), skip_front=True, time_frac=0.02)
            case 3:
                sketch = drawPNG(json.load(open(sketch_path)), skip_front=False, time_frac=0.05)
            case 4:
                sketch = drawPNG(json.load(open(sketch_path)), add_stroke=True)
            case _:
                sketch = drawPNG(json.load(open(sketch_path)))

        # c = self.rng.choice([True, False])
        # if c:
        #     sketch = drawPNG(json.load(open(sketch_path)), skip_front=True, time_frac=0.03)
        # else:
        #     sketch = drawPNG(json.load(open(sketch_path)))

        # sketch = drawPNG(json.load(open(sketch_path)))

        sketch = Image.fromarray(sketch)
        # sketch = ImageOps.pad(sketch, (224, 224), method=Resampling.BILINEAR)

        image = Image.open(image_path)
        # image = ImageOps.pad(image, (224, 224), method=Resampling.BILINEAR)

        negative = Image.open(negative_path)
        # negative = ImageOps.pad(negative, (224, 224), method=Resampling.BILINEAR)

        if self.transforms_sketch:
            sketch = self.transforms_sketch(sketch)

        if self.transforms_image:
            image = self.transforms_image(image)
            negative = self.transforms_image(negative)

        return sketch, image, negative


class DatasetTest(Dataset):
    def __init__(self, root, sketch, num_of_users=100, transforms=None):
        if num_of_users > 100:
            raise ValueError('num_of_users > 100')

        self.root = root

        with open(os.path.join(self.root, "..", 'val_unseen_user.txt'), 'r') as f:
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
        self.sketch = sketch

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if self.sketch:
            img_path = os.path.join(self.root, self.files[idx] + ".json")
            img = drawPNG(json.load(open(img_path)))
            img = Image.fromarray(img)
            # img = ImageOps.pad(img, (224, 224), method=Resampling.BILINEAR)
        else:
            img_path = os.path.join(self.root, self.files[idx] + ".jpg")
            img = Image.open(img_path)
            # img = ImageOps.pad(img, (224, 224), method=Resampling.BILINEAR)

        if self.transforms is not None:
            img = self.transforms(img)

        return img
