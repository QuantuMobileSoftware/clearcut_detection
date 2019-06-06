import collections

import pandas as pd
import torch
from torch import nn
from catalyst.dl.callbacks import InferCallback, CheckpointCallback, EarlyStoppingCallback, DiceCallback
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.utils import UtilsFactory

from datasets import create_loaders
from losses import BCE_Dice_Loss
from utils import get_model
from params import args


CHANNELS_COUNT = 9


def main():
    model = get_model('fpn50')
    print('Loading model')
    model.encoder.conv1 = nn.Conv2d(CHANNELS_COUNT, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model, device = UtilsFactory.prepare_model(model)

    train_df = pd.read_csv(args.train_df).to_dict('records')
    val_df = pd.read_csv(args.val_df).to_dict('records')

    loaders = create_loaders(train_df, val_df)

    criterion = BCE_Dice_Loss()
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
        ],
        logdir=args.logdir,
        num_epochs=args.epochs,
        verbose=True
    )

    infer_loader = collections.OrderedDict([('infer', loaders['valid'])])
    runner.infer(
        model=model,
        loaders=infer_loader,
        callbacks=[
            CheckpointCallback(resume=f'{args.logdir}/checkpoints/best.pth'),
            InferCallback()
        ],
    )


if __name__ == '__main__':
    main()
