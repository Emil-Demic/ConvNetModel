import random
import tqdm
import torch
import numpy as np
from info_nce import InfoNCE

from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Resize, Normalize, Compose, ToImage, ToDtype, RGB, Grayscale

from config import args
from data import DatasetFSCOCO
from model import SbirModel
from utils import calculate_accuracy_alt, compute_view_specific_distance, calculate_accuracy

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)

transforms = Compose([
    RGB(),
    Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
    ToImage(),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    Grayscale(num_output_channels=3),
])

dataset_train = DatasetFSCOCO("fscoco", "train", args.users, transforms, transforms)
dataset_val = DatasetFSCOCO("fscoco", "val", args.users, transforms, transforms)

dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size * 3, shuffle=False)

model = SbirModel(args.model)
if args.cuda:
    model.cuda()

optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
iters = 70 * args.users * 3 / args.batch_size
scheduler = LinearLR(optimizer, start_factor=1./10., total_iters=iters)

loss_fn = InfoNCE(negative_mode="unpaired", temperature=0.05)


best_res = 0
no_improve = 0
for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader_train):
        if args.cuda:
            data = [d.cuda() for d in data]

        output = model(data)

        loss = loss_fn(output[0], output[1])

        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        scheduler.step()

        if i % 3 == 2 or i == len(dataloader_train) - 1:
            print(f'[{epoch:03d}, {i:03d}] loss: {running_loss / 3:0.5f}')
            running_loss = 0.0

    print(f"lr: {optimizer.state_dict()['param_groups'][0]['lr']}")

    with torch.no_grad():
        model.eval()

        sketch_output = []
        image_output = []
        for data in tqdm.tqdm(dataloader_val):
            if args.cuda:
                data = [d.cuda() for d in data]

            output = model(data)
            sketch_output.append(output[0].cpu().numpy())
            image_output.append(output[1].cpu().numpy())

        sketch_output = np.concatenate(sketch_output)
        image_output = np.concatenate(image_output)

        dis = compute_view_specific_distance(sketch_output, image_output)

        top1, top5, top10 = calculate_accuracy(dis, dataset_val.get_file_names())
        print("top1, top5, top10:", top1, top5, top10)

        top1, top5, top10, meanK = calculate_accuracy_alt(sketch_output, image_output)
        num = sketch_output.shape[0]
        print("top1, top5, top10, meanK:", top1, top5, top10, meanK)
        print(str(epoch + 1) + ':  top1: ' + str(top1 / float(num)))
        print(str(epoch + 1) + ':  top5: ' + str(top5 / float(num)))
        print(str(epoch + 1) + ': top10: ' + str(top10 / float(num)))

        if top10 > best_res:
            best_res = top10
        else:
            no_improve += 1
            if no_improve == 3:
                break

if args.save:
    torch.save(model.state_dict(), "model.pth")
