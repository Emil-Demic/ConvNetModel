import os
import time

import numpy as np
import torch
import tqdm
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Resize, Normalize, Compose, ToImage, ToDtype, RGB

from model import TripletModel

img_paths = []

with open("../fscoco/val_unseen_user.txt", "r") as f:
    val_files = f.readlines()

val_files = [int(i) for i in val_files]
for i in range(1, 101):
    path_to_dir = os.path.join("../fscoco", "images", str(i))
    files_in_dir = os.listdir(path_to_dir)
    for filename in files_in_dir:
        if int(filename.split(".")[0]) in val_files:
            img_paths.append(os.path.join(path_to_dir, filename))

# folders = os.listdir('../imagenet-a')
# folders.remove("README.txt")
# for folder in folders[:20]:
#     path_to_dir = os.path.join("../imagenet-a", folder)
#     files_in_dir = os.listdir(path_to_dir)
#     for filename in files_in_dir:
#         img_paths.append(os.path.join(path_to_dir, filename))


model = TripletModel("convnext")
model.load_state_dict(torch.load('model_unseen.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

transforms = Compose([
    RGB(),
    Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
    ToImage(),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_embeds = []

start_time = time.time()

with torch.no_grad():
    for img_path in tqdm.tqdm(img_paths):
        img = Image.open(img_path)
        img = transforms(img).unsqueeze(0)
        img_embeds.append(model.get_embedding(img).numpy()[0])

elapsed_time = time.time() - start_time
print("Time: ", elapsed_time)

img_embeds = np.stack(img_embeds)
img_paths = np.stack(img_paths)
np.save("unseen_emb.npy", img_embeds)
np.save("unseen_paths.npy", img_paths)



