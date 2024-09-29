import random
import tqdm
import torch
import numpy as np
from info_nce import InfoNCE

from torch.nn import TripletMarginLoss
from torch.optim import Adam, lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Resize, Normalize, Compose, ToImage, ToDtype, RGB

from config import args
from data import DatasetTrain, DatasetTest
from model import TripletModel
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
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_train = DatasetTrain("fscoco", args.users, transforms, transforms)
dataset_test_sketch = DatasetTest("fscoco/raster_sketches", args.users, transforms)
dataset_test_image = DatasetTest("fscoco/images", args.users, transforms)

dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
dataloader_test_sketch = DataLoader(dataset_test_sketch, batch_size=args.batch_size * 3, shuffle=False)
dataloader_test_image = DataLoader(dataset_test_image, batch_size=args.batch_size * 3, shuffle=False)

model = TripletModel(args.model)
if args.cuda:
    model.cuda()

optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, args.lr_scheduler_step, gamma=0.1, last_epoch=-1)

loss_fn = InfoNCE(negative_mode="unpaired", temperature=0.05)
# loss_fn = TripletMarginLoss(margin=0.2)
# if args.cuda:
#     loss_fn.cuda()

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(dataloader_train):
        optimizer.zero_grad()
        if args.cuda:
            data = [d.cuda() for d in data]

        output = model(data)

        mask = ~(output[1] == output[2]).all(dim=1)

        loss = loss_fn(output[0], output[1], output[2][mask])

        # loss = loss_fn(output[0], output[1], output[2])

        running_loss += loss.item()
        loss.backward()
        optimizer.step()

        if i % 5 == 4:
            print(f'[{epoch:03d}, {i:03d}] loss: {running_loss / 5:0.5f}')
            running_loss = 0.0

    print(f"lr: {optimizer.state_dict()['param_groups'][0]['lr']}")
    scheduler.step()

    with torch.no_grad():
        model.eval()

        sketch_output = []
        for data in tqdm.tqdm(dataloader_test_sketch):
            if args.cuda:
                data = data.cuda()
            out = model.get_embedding(data)
            sketch_output.append(out.cpu().numpy())

        image_output = []
        for data in tqdm.tqdm(dataloader_test_image):
            if args.cuda:
                data = data.cuda()
            out = model.get_embedding(data)
            image_output.append(out.cpu().numpy())

        sketch_output = np.concatenate(sketch_output)
        image_output = np.concatenate(image_output)

        dis = compute_view_specific_distance(sketch_output, image_output)

        top1, top5, top10 = calculate_accuracy(dis, dataset_test_image.get_file_names())
        print("top1, top5, top10:", top1, top5, top10)

        top1, top5, top10, meanK = calculate_accuracy_alt(sketch_output, image_output)
        num = sketch_output.shape[0]
        print("top1, top5, top10, meanK:", top1, top5, top10, meanK)
        print(str(epoch + 1) + ':  top1: ' + str(top1 / float(num)))
        print(str(epoch + 1) + ':  top5: ' + str(top5 / float(num)))
        print(str(epoch + 1) + ': top10: ' + str(top10 / float(num)))

if args.save:
    torch.save(model.state_dict(), "model.pth")
