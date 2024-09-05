import json
import os
import random

from PIL import Image, ImageOps
from PIL.Image import Resampling
from torch.utils.data import Dataset

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

        self.strokes_to_remove = 0.0

        random.seed(42)


    def increase_strokes_to_remove(self):
        self.strokes_to_remove += 0.01

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.root, "raw_data", self.files[idx] + ".json")
        image_path = os.path.join(self.root, "images", self.files[idx] + ".jpg")

        negative_idx = random.randint(0, len(self.files) - 1)
        while negative_idx == idx:
            negative_idx = random.randint(0, len(self.files) - 1)

        negative_path = os.path.join(self.root, "images", self.files[negative_idx] + ".jpg")

        remove_strokes = random.choice([True, False])
        if remove_strokes:
            remove_front = random.choice([True, False])
            if remove_front:
                sketch = drawPNG(json.load(open(sketch_path)), time_frac=self.strokes_to_remove, skip_front=True)
            else:
                sketch = drawPNG(json.load(open(sketch_path)), time_frac=self.strokes_to_remove * 4.0, skip_front=False)
        else:
            sketch = drawPNG(json.load(open(sketch_path)))
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

# class DatasetTrain(Dataset):
#     def __init__(self, root, transforms_sketch=None, transforms_image=None):
#         self.root = root
#         self.sketches = os.listdir(os.path.join(self.root, "sketch", "Image"))
#         self.images = os.listdir(os.path.join(self.root, "image", "Image"))
#         self.transforms_sketch = transforms_sketch
#         self.transforms_image = transforms_image
#         random.seed(42)
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         sketch_path = os.path.join(self.root, "sketch", "Image", self.sketches[idx])
#         image_path = os.path.join(self.root, "image", "Image", self.images[idx])
#
#         negative_idx = random.randint(0, len(self.sketches) - 1)
#         while negative_idx == idx:
#             negative_idx = random.randint(0, len(self.sketches) - 1)
#
#         negative_path = os.path.join(self.root, "image", "Image", self.images[negative_idx])
#
#         sketch = Image.open(sketch_path)
#         image = Image.open(image_path).convert('RGB')
#         negative = Image.open(negative_path).convert('RGB')
#
#         if self.transforms_sketch:
#             sketch = self.transforms_sketch(sketch)
#
#         if self.transforms_image:
#             image = self.transforms_image(image)
#             negative = self.transforms_image(negative)
#
#         return sketch, image, negative


# class DatasetTest2(Dataset):
#     def __init__(self, img_dir, transforms=None):
#         self.img_dir = img_dir
#         self.images = os.listdir(img_dir)
#         self.transforms = transforms
#
#     def __len__(self):
#         return len(self.images)
#
#     def __getitem__(self, idx):
#         img_path = os.path.join(self.img_dir, self.images[idx])
#         img = Image.open(img_path)
#         if self.transforms is not None:
#             img = self.transforms(img)
#
#         return img


