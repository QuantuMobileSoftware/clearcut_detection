import argparse
import collections
import os

import pandas as pd
import torch
from catalyst.dl.callbacks import DiceCallback, CheckpointCallback, InferCallback
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.utils import UtilsFactory

from clearcut_research.pytorch.dataset import Dataset
from clearcut_research.pytorch.losses import BCE_Dice_Loss
from clearcut_research.pytorch.models.utils import get_model, set_random_seed


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--batch_size', type=int, default=8)
    arg('--num_workers', type=int, default=4)
    arg('--epochs', '-e', type=int, default=100)
    arg('--folds', '-f', type=int)

    arg('--logdir', default='../logs')
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
    for fold in range(args.folds):
        model = get_model(args.network)

        print("Loading model")
        model, device = UtilsFactory.prepare_model(model)
        train_df = pd.read_csv(os.path.join(args.dataset_path, f'train{fold}.csv')).to_dict('records')
        val_df = pd.read_csv(os.path.join(args.dataset_path, f'val{fold}.csv')).to_dict('records')

        ds = Dataset(args.channels, args.dataset_path, args.image_size, args.batch_size, args.num_workers)
        loaders = ds.create_loaders(train_df, val_df)

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
    args = parse_args()
    train(args)
