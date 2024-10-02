import os

from PIL import Image
from torch.utils.data import Dataset
from config import args


class DatasetFSCOCO(Dataset):
    def __init__(self, root, mode="train", num_of_users=100, transforms_sketch=None, transforms_image=None):
        if num_of_users > 100:
            raise ValueError('num_of_users > 100')

        self.root = root

        val_path = "val_unseen_user.txt" if args.val_unseen else "val_normal.txt"
        with open(os.path.join(self.root, val_path), 'r') as f:
            lines = f.readlines()
            val_ids = set(map(int, lines))

        self.files = []
        for i in range(1, num_of_users + 1):
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
        return self.files


class DatasetTest(Dataset):
    def __init__(self, root, num_of_users=100, transforms=None):
        if num_of_users > 100:
            raise ValueError('num_of_users > 100')

        self.root = root

        val_path = "val_unseen_user.txt" if args.val_unseen else "val_normal.txt"
        with open(os.path.join(self.root, "..", val_path), 'r') as f:
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
