import sys

sys.path.append("../..")

import torch
from catalyst.dl.metrics import dice
from catalyst.dl.utils import UtilsFactory
from torch.nn import functional as F

from datasets import create_loaders
from losses import BCE_Dice_Loss
from models.utils import get_model
from params import args
from statistics import mean


def main():
    model = get_model(args.network)

    print("Loading model")
    model, device = UtilsFactory.prepare_model(model)

    loaders = create_loaders()

    criterion = BCE_Dice_Loss(bce_weight=0.2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.3)

    best_valid_dice = -1
    best_epoch = -1

    for epoch in range(args.epochs):

        train_losses = []
        train_dices = []
        valid_losses = []
        valid_dices = []

        segmentation_weight = 0.8

        for i, data in enumerate(loaders["train"], 0):
            input = data["features"]
            targets = data["targets"]

            x = input.to(device)

            mask, season = targets

            mask = mask.to(device)
            season = season.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            season_prediction, mask_prediction = model(x)

            segmentation_loss = criterion(mask_prediction, mask)
            season_loss = F.binary_cross_entropy(season_prediction, season)

            dice_score = dice(mask_prediction, mask).item()

            print("Bce_dice_loss:", round(segmentation_loss.mean().item(), 3),
                  "Bce_season:", round(season_loss.mean().item(), 3),
                  "Dice:", round(dice_score, 3), end='\r', flush=True)

            loss = segmentation_loss * segmentation_weight + season_loss * (1 - segmentation_weight)

            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_dices.append(dice_score)

        print()
        print("Train loss:", round(mean(train_losses), 3))
        print("Train dice:", round(mean(train_dices), 3))

        with torch.no_grad():
            for valid_idx, valid_data in enumerate(loaders["valid"], 0):
                input = valid_data["features"]
                targets = valid_data["targets"]

                x = input.to(device)

                mask, season = targets

                mask = mask.to(device)
                season = season.to(device)

                # forward + backward + optimize
                season_prediction, mask_prediction = model(x)

                segmentation_loss = criterion(mask_prediction, mask)
                season_loss = F.binary_cross_entropy(season_prediction, season)

                loss = segmentation_loss * segmentation_weight + season_loss * (1 - segmentation_weight)

                valid_losses.append(loss.item())
                valid_dices.append(dice(mask_prediction, mask).item())

            dice_mean = round(mean(valid_dices), 3)
            print("Valid loss:", round(mean(valid_losses), 3), "Valid dice:", dice_mean)

            if dice_mean > best_valid_dice:
                best_valid_dice = dice_mean
                best_epoch = epoch
                torch.save(model.state_dict(), 'weights/epoch{0}.pth'.format(epoch))

        scheduler.step()
        print("Epoch {0} ended".format(epoch))

    print("Best epoch: ", best_epoch, "with dice: ", best_valid_dice)


if __name__ == '__main__':
    main()
