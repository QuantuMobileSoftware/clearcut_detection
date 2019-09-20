import argparse
import collections

import torch
import pandas as pd

from catalyst.dl.callbacks import CheckpointCallback, InferCallback
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.utils import UtilsFactory

from clearcut_research.pytorch.models.utils import get_model, set_random_seed
from clearcut_research.pytorch.losses import MultiClass_Dice_Loss
from clearcut_research.pytorch.models.multiclass_prediction.multiclass_dataset import MulticlassDataset
from clearcut_research.pytorch.models.multiclass_prediction.multiclass_dice_callback import MultiClassDiceCallback


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
    model = get_model('fpn50_multiclass')

    print("Loading model")
    model, device = UtilsFactory.prepare_model(model)

    train_df = pd.read_csv(args.train_df).to_dict('records')
    val_df = pd.read_csv(args.val_df).to_dict('records')

    ds = MulticlassDataset(args.channels, args.dataset_path, args.image_size, args.batch_size, args.num_workers)
    loaders = ds.create_loaders(train_df, val_df)

    criterion = MultiClass_Dice_Loss()
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
            MultiClassDiceCallback()
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
    args = parse_args()
    train(args)
