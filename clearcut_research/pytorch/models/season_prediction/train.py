import argparse
import os

import torch
import pandas as pd

from catalyst.dl.metrics import dice
from catalyst.dl.utils import UtilsFactory
from torch.nn import functional as F

from clearcut_research.pytorch.models.season_prediction.season_dataset import SeasonDataset
from clearcut_research.pytorch.losses import BCE_Dice_Loss
from clearcut_research.pytorch.models.utils import get_model, set_random_seed
from statistics import mean


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--batch_size', type=int, default=8)
    arg('--num_workers', type=int, default=4)
    arg('--epochs', '-e', type=int, default=100)

    arg('--logdir', default='../logs')
    arg('--train_df', '-td', default='../data/train_df.csv')
    arg('--val_df', '-vd', default='../data/val_df.csv')
    arg('--dataset_path', '-dp', default='../data/input', help='Path to the data')

    arg('--image_size', '-is', type=int, default=224)
    arg('--network', '-n', default='unet50')
    arg(
        '--channels', '-ch',
        default=[
            'rgb', 'ndvi', 'ndvi_color',
            'b2', 'b3', 'b4', 'b8'
        ], nargs='+', help='Channels list')

    return parser.parse_args()


def train(args):
    set_random_seed(42)
    model = get_model('fpn50_season')

    print("Loading model")
    model, device = UtilsFactory.prepare_model(model)

    train_df = pd.read_csv(args.train_df).to_dict('records')
    val_df = pd.read_csv(args.val_df).to_dict('records')

    ds = SeasonDataset(args.channels, args.dataset_path, args.image_size, args.batch_size, args.num_workers)
    loaders = ds.create_loaders(train_df, val_df)

    criterion = BCE_Dice_Loss(bce_weight=0.2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.3)

    best_valid_dice = -1
    best_epoch = -1
    best_accuracy = -1

    for epoch in range(args.epochs):
        segmentation_weight = 0.8

        train_iter(loaders['train'], model, device, criterion, optimizer, segmentation_weight)
        dice_mean, valid_accuracy = valid_iter(loaders['valid'], model, device, criterion, segmentation_weight)

        if dice_mean > best_valid_dice:
            best_valid_dice = dice_mean
            best_epoch = epoch
            best_accuracy = valid_accuracy
            os.makedirs(f'{args.logdir}/weights', exist_ok=True)
            torch.save(model.state_dict(), f'{args.logdir}/weights/epoch{epoch}.pth')

        scheduler.step()
        print("Epoch {0} ended".format(epoch))

    print("Best epoch: ", best_epoch, "with dice ", best_valid_dice, "and season prediction accuracy", best_accuracy)


def train_iter(train_loader, model, device, criterion, optimizer, segmentation_weight):
    train_losses = []
    train_dices = []
    train_size = 0
    train_predicted = 0

    for i, data in enumerate(train_loader, 0):
        input = data["features"]
        targets = data["targets"]
        x = input.to(device)
        mask, season = targets
        mask = mask.to(device)
        season = season.to(device)

        optimizer.zero_grad()
        season_prediction, mask_prediction = model(x)

        segmentation_loss = criterion(mask_prediction, mask)
        season_loss = F.binary_cross_entropy(season_prediction, season)
        loss = segmentation_loss * segmentation_weight + season_loss * (1 - segmentation_weight)
        loss.backward()
        optimizer.step()

        season_prediction = season_prediction > 0.5
        train_size += season_prediction.size()[0]
        train_predicted += torch.sum((season_prediction == season.byte())).item()

        dice_score = dice(mask_prediction, mask).item()
        train_losses.append(loss.item())
        train_dices.append(dice_score)

        print("Bce_dice_loss: {:.3f}".format(segmentation_loss.mean().item()), '\t',
              "Bce_season: {:.3f}".format(season_loss.mean().item()), '\t',
              "Dice: {:.3f}".format(dice_score),
              end='\r', flush=True,
              )

    print()
    train_accuracy = train_predicted / train_size

    print("Train:")
    print("Loss:", round(mean(train_losses), 3), "Dice:", round(mean(train_dices), 3), "Accuracy:",
          round(train_accuracy, 3))


def valid_iter(valid_loader,  model, device, criterion, segmentation_weight):
    valid_losses = []
    valid_dices = []
    valid_size = 0
    valid_predicted = 0

    with torch.no_grad():
        for valid_idx, valid_data in enumerate(valid_loader, 0):
            input = valid_data["features"]
            targets = valid_data["targets"]
            x = input.to(device)
            mask, season = targets
            mask = mask.to(device)
            season = season.to(device)

            season_prediction, mask_prediction = model(x)

            segmentation_loss = criterion(mask_prediction, mask)
            season_loss = F.binary_cross_entropy(season_prediction, season)
            loss = segmentation_loss * segmentation_weight + season_loss * (1 - segmentation_weight)

            season_prediction = season_prediction > 0.5

            valid_size += season_prediction.size()[0]
            valid_predicted += torch.sum((season_prediction == season.byte())).item()

            valid_losses.append(loss.item())
            valid_dices.append(dice(mask_prediction, mask).item())

        dice_mean = round(mean(valid_dices), 3)
        valid_accuracy = valid_predicted / valid_size
        print("Validation:")
        print("Loss:", round(mean(valid_losses), 3), "Dice:", dice_mean, "Accuracy:", round(valid_accuracy, 3))

    return dice_mean, valid_accuracy


if __name__ == '__main__':
    args = parse_args()
    train(args)
