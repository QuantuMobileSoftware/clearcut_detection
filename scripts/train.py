import os
import collections
import pandas as pd

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from catalyst.dl.utils import UtilsFactory
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.callbacks import InferCallback, CheckpointCallback

from PIL import Image
from torch.nn import functional

from .params import args
from .losses import dice_loss
from .unet_models import UNetWithResnet50Encoder


def bce_dice(input, target, weight=0.5):
    bce_loss = nn.BCEWithLogitsLoss()
    return bce_loss(input, target) * weight + dice_loss(input, target) * (1 - weight)


def get_image(image_name):
    dataset_path = args.dataset_path
    img_path = os.path.join(dataset_path, 'images', image_name + '.png')
    mask_path = os.path.join(dataset_path, 'masks', image_name + '.png')

    img = Image.open(img_path)
    mask = Image.open(mask_path)

    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    img_tensor = data_transform(img)
    mask_tensor = transforms.ToTensor()(mask)

    return {'features': img_tensor, 'targets': mask_tensor}


class BCEDiceLoss(torch.nn.Module):

    def __init__(self):
        super(BCEDiceLoss, self).__init__()

    @staticmethod
    def calc_loss(prediction, target, bce_weight=0.5):
        bce = functional.binary_cross_entropy_with_logits(prediction, target)

        prediction = torch.sigmoid(prediction)
        dice = dice_loss(prediction, target)

        loss = bce * bce_weight + dice * (1 - bce_weight)

        return loss

    def forward(self, x, y):
        return self.calc_loss(x, y)


def main():
    model = UNetWithResnet50Encoder(n_classes=1)

    model, device = UtilsFactory.prepare_model(model)

    train_df = pd.read_csv(args.train_df)
    val_df = pd.read_csv(args.val_df)

    train_loader = UtilsFactory.create_loader(
        train_df['image_name'],
        open_fn=get_image,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True)

    valid_loader = UtilsFactory.create_loader(
        val_df['image_name'],
        open_fn=get_image,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True)

    loaders = collections.OrderedDict()
    loaders['train'] = train_loader
    loaders['valid'] = valid_loader

    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.3)

    # model runner
    runner = SupervisedRunner()

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        logdir=args.logdir,
        num_epochs=args.epochs,
        verbose=True
    )

    infer_loader = collections.OrderedDict([('infer', loaders['valid'])])
    runner.infer(
        model=model,
        loaders=infer_loader,
        callbacks=[
            CheckpointCallback(
                resume=f'{args.logdir}/checkpoints/best.pth'),
            InferCallback()
        ],
    )


if __name__ == '__main__':
    main()

# class CleanCutDataset(Dataset):
#     def __init__(self, csv_file, root_dir, transform=None):
#         self.dataset_names = pd.read_csv(csv_file)
#         self.root_dir = root_dir
#         self.transform = transform
#
#     def __len__(self):
#         return len(self.dataset_names)
#
#     def __getitem__(self, idx):
#
#         img_path = os.path.join(self.root_dir, "images", self.dataset_names["image_name"][idx] + ".png")
#         mask_path = os.path.join(self.root_dir, "masks", self.dataset_names["image_name"][idx] + ".png")
#
#         image = Image.open(img_path)
#         mask = Image.open(mask_path)
#
#         sample = {'image': image, 'mask': mask}
#
#         if self.transform:
#             sample = self.transform(sample)
#
#         return sample
