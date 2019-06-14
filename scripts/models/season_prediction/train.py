import torch
from catalyst.dl.metrics import dice
from catalyst.dl.utils import UtilsFactory
from torch.nn import functional as F

from .datasets import create_loaders
from losses import BCE_Dice_Loss
from models.utils import get_model
from params import args


def main():
    model = get_model(args.network)

    print("Loading model")
    model, device = UtilsFactory.prepare_model(model)

    loaders = create_loaders()

    criterion = BCE_Dice_Loss(bce_weight=0.2)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.3)

    validate_every = 1
    iteration = 0
    for epoch in range(args.epochs):  # loop over the dataset multiple times

        # running_loss = 0.0
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

            print(segmentation_loss.mean(), season_loss.mean(), dice(mask_prediction, mask))

            loss = segmentation_loss + season_loss

            loss.backward()
            optimizer.step()

            print()

        # for valid_idx, valid_data in enumerate(loaders["valid"], 0):
        #     input, label = valid_data
        #
        #     valid_model = model.eval()
        #
        #     # forward + backward + optimize
        #     season, mask = valid_model(input)
        #     segmentation_loss = criterion(mask, labels)
        #
        #     season_loss = F.binary_cross_entropy_with_logits(season, labels)
        #     print(segmentation_loss, season_loss)

        print("Epoch ended")

    # # model runner
    # runner = SupervisedRunner()
    #
    # # model training
    # runner.train(
    #     model=model,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     scheduler=scheduler,
    #     loaders=loaders,
    #     callbacks=[
    #         DiceCallback()
    #         # EarlyStoppingCallback(patience=20, min_delta=0.01)
    #     ],
    #     logdir=args.logdir,
    #     num_epochs=args.epochs,
    #     verbose=True
    # )
    #
    # infer_loader = collections.OrderedDict([("infer", loaders["valid"])])
    # runner.infer(
    #     model=model,
    #     loaders=infer_loader,
    #     callbacks=[
    #         CheckpointCallback(
    #             resume=f"{args.logdir}/checkpoints/best.pth"),
    #         InferCallback()
    #     ],
    # )


if __name__ == '__main__':
    main()
