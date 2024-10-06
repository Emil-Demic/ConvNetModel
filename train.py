import tqdm
import torch

from info_nce import InfoNCE
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import args
from model import SbirModel
from utils import compute_view_specific_distance, calculate_results, seed_everything
from data import create_datasets

seed_everything()

dataset_train, dataset_val = create_datasets(args.dataset, args.root)

dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True)
dataloader_val = DataLoader(dataset_val, batch_size=args.batch_size * 3, shuffle=False)

model = SbirModel(args.backbone)
model.load_state_dict(torch.load("full_model.pth", weights_only=True))
if args.cuda:
    model.cuda()


optimizer = Adam(model.parameters(), lr=args.lr)

loss_fn = InfoNCE(negative_mode="unpaired", temperature=args.temp)

best_res = 0
best_top1 = 0
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

        if i % 5 == 4:
            print(f'[{epoch:03d}, {i:03d}] loss: {running_loss / 5  :0.5f}')
            running_loss = 0.0

    with torch.no_grad():
        model.eval()

        sketch_output = []
        image_output = []
        for data in tqdm.tqdm(dataloader_val):
            if args.cuda:
                data = [d.cuda() for d in data]

            output = model(data)
            sketch_output.append(output[0].cpu())
            image_output.append(output[1].cpu())

        sketch_output = torch.concatenate(sketch_output)
        image_output = torch.concatenate(image_output)

        if args.dataset.lower() == 'shoev2' or args.dataset.lower() == 'chairv2':
            image_output = torch.unique_consecutive(image_output, dim=0)

        dis = compute_view_specific_distance(sketch_output.numpy(), image_output.numpy())

        print(f"EPOCH {str(epoch)}:")
        top1, top5, top10 = calculate_results(dis, dataset_val.get_file_names(), dataset_val.get_file_paths(),
                                              dataset_val.get_file_map())

        if top5 > best_res:
            no_improve = 0
            best_res = top5
            best_top1 = top1
            if args.save:
                torch.save(model.state_dict(), f"E{epoch}_model.pth")
        else:
            if args.save and top1 > best_top1 and top5 == best_res:
                best_top1 = top1
                torch.save(model.state_dict(), f"E{epoch}_model.pth")
            no_improve += 1
            if no_improve == 2:
                print("top10 metric has not improved for 2 epochs. Ending training.")
                break
