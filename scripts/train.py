import collections
import os

import torch
from catalyst.dl.callbacks import DiceCallback, CheckpointCallback, InferCallback
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.utils import UtilsFactory

from datasets import create_loaders
from losses import BCE_Dice_Loss
from models.utils import get_model
from params import args


def main():
    for fold in range(args.folds):
        model = get_model(args.network)

        print("Loading model")
        model, device = UtilsFactory.prepare_model(model)
        loaders = create_loaders(train_df=os.path.join(args.train_df, f'train{fold}.csv'),
                                 val_df=os.path.join(args.val_df, f'val{fold}.csv'))

        criterion = BCE_Dice_Loss(bce_weight=0.2)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.3)

        # model runner
        runner = SupervisedRunner()

        save_path = os.path.join(
            args.logdir,
            f'fold{fold}'
        )

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
            logdir=save_path,
            num_epochs=args.epochs,
            verbose=True
        )

        infer_loader = collections.OrderedDict([("infer", loaders["valid"])])
        runner.infer(
            model=model,
            loaders=infer_loader,
            callbacks=[
                CheckpointCallback(
                    resume=f'{save_path}/checkpoints/best.pth'),
                InferCallback()
            ],
        )

        print(f'Fold {fold} ended')


if __name__ == '__main__':
    main()
