import collections

import pandas as pd
import torch
from catalyst.dl.callbacks import InferCallback, CheckpointCallback, EarlyStoppingCallback, DiceCallback
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.utils import UtilsFactory

from datasets import create_loaders
from losses import BCE_Dice_Loss
from models.utils import get_model
from params import args


def main():
    model = get_model('resnet50')
    print('Loading model')
    model, device = UtilsFactory.prepare_model(model)

    train_df = pd.read_csv(args.train_df, index_col=0).to_dict('records')
    val_df = pd.read_csv(args.val_df, index_col=0).to_dict('records')

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
            DiceCallback(),
            EarlyStoppingCallback(patience=2, min_delta=0.01)
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
