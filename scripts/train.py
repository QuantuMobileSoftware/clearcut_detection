import collections

import torch
from catalyst.dl.callbacks import InferCallback, CheckpointCallback, EarlyStoppingCallback, DiceCallback
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.utils import UtilsFactory

from datasets import create_loaders
from losses import BCE_Dice_Loss, Bootstrapped_BCE_Dice_Loss
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

    # model runner
    runner = SupervisedRunner()

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[
            DiceCallback()
            # EarlyStoppingCallback(patience=20, min_delta=0.01)
        ],
        logdir=args.logdir,
        num_epochs=args.epochs,
        verbose=True
    )

    infer_loader = collections.OrderedDict([("infer", loaders["valid"])])
    runner.infer(
        model=model,
        loaders=infer_loader,
        callbacks=[
            CheckpointCallback(
                resume=f"{args.logdir}/checkpoints/best.pth"),
            InferCallback()
        ],
    )


if __name__ == '__main__':
    main()
