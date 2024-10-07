import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Resize, Normalize, Compose, ToImage, ToDtype, RGB

from config import args


def create_datasets(dataset, root):
    transforms = Compose([
            RGB(),
            Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    match dataset.lower():
        case "fscoco":
            dataset_train = DatasetFSCOCO(root, mode="train", transforms_sketch=transforms,
                                          transforms_image=transforms)
            dataset_val = DatasetFSCOCO(root, mode="val", transforms_sketch=transforms,
                                        transforms_image=transforms)
            return dataset_train, dataset_val

        case "chairv2":
            dataset_train = DatasetQMUL(root, mode="train", transforms_sketch=transforms,
                                        transforms_image=transforms)
            dataset_val = DatasetQMUL(root, mode="val", transforms_sketch=transforms,
                                      transforms_image=transforms)
            return dataset_train, dataset_val

        case "shoev2":
            dataset_train = DatasetQMUL(root, mode="train", transforms_sketch=transforms,
                                        transforms_image=transforms)
            dataset_val = DatasetQMUL(root, mode="val", transforms_sketch=transforms,
                                      transforms_image=transforms)
            return dataset_train, dataset_val

        case _:
            raise ValueError(f"Unknown dataset {dataset}")


class DatasetFSCOCO(Dataset):
    def __init__(self, root, mode="train", transforms_sketch=None, transforms_image=None):

        self.root = root

        val_path = "val_unseen_user.txt" if args.val_unseen else "val_normal.txt"
        with open(os.path.join(self.root, val_path), 'r') as f:
            lines = f.readlines()
            val_ids = set(map(int, lines))

        self.files = []
        for i in range(1, 101):
            file_names = os.listdir(os.path.join(root, "images", str(i)))
            file_names = [file.split('.')[0] for file in file_names]
            if mode == "train":
                file_names = [file for file in file_names if int(file) not in val_ids]
            else:
                file_names = [file for file in file_names if int(file) in val_ids]
            for file_name in file_names:
                self.files.append(os.path.join(str(i), file_name + ".jpg"))

        self.transforms_sketch = transforms_sketch
        self.transforms_image = transforms_image

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sketch_path = os.path.join(self.root, "raster_sketches", self.files[idx])
        image_path = os.path.join(self.root, "images", self.files[idx])

        sketch = Image.open(sketch_path)
        image = Image.open(image_path)

        if self.transforms_sketch:
            sketch = self.transforms_sketch(sketch)

        if self.transforms_image:
            image = self.transforms_image(image)

        return sketch, image

    def get_file_names(self):
        return self.files, self.files

    def get_file_paths(self):
        return os.path.join(self.root, "raster_sketches"), os.path.join(self.root, "images")

    def get_file_map(self):
        file_map = {}
        for i in range(len(self.files)):
            file_map[i] = i
        return file_map


class DatasetQMUL(Dataset):
    def __init__(self, root, mode="train", transforms_sketch=None, transforms_image=None):
        self.root = root
        self.mode = mode

        if mode == "train":
            self.files_sketches = sorted(os.listdir(os.path.join(root, "trainA")))
            self.files_imgs = sorted(os.listdir(os.path.join(root, "trainB")))
        else:
            self.files_sketches = sorted(os.listdir(os.path.join(root, "testA")))
            self.files_imgs = sorted(os.listdir(os.path.join(root, "testB")))

        self.transforms_sketch = transforms_sketch
        self.transforms_image = transforms_image

    def __len__(self):
        return len(self.files_sketches)

    def __getitem__(self, idx):
        image_name = self.files_sketches[idx].split("_")[:-1]
        image_name = "_".join(image_name) + ".png"

        if self.mode == "train":
            sketch_path = os.path.join(self.root, "trainA", self.files_sketches[idx])
            image_path = os.path.join(self.root, "trainB", image_name)
        else:
            sketch_path = os.path.join(self.root, "testA", self.files_sketches[idx])
            image_path = os.path.join(self.root, "testB", image_name)

        sketch = Image.open(sketch_path)
        image = Image.open(image_path)

        if self.transforms_sketch:
            sketch = self.transforms_sketch(sketch)

        if self.transforms_image:
            image = self.transforms_image(image)

        return sketch, image

    def get_file_names(self):
        return self.files_sketches, self.files_imgs

    def get_file_paths(self):
        return os.path.join(self.root, "testA"), os.path.join(self.root, "testB")

    def get_file_map(self):
        file_map = {}
        for i, file in enumerate(self.files_sketches):
            name = file.split("_")[:-1]
            name = "_".join(name) + ".png"
            file_map[i] = self.files_imgs.index(name)
        return file_map
