import argparse
import collections

import pandas as pd
import torch
from catalyst.dl.callbacks import InferCallback, CheckpointCallback
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.utils import UtilsFactory
from torch.nn import MSELoss

from clearcut_research.pytorch.models.autoencoder.autoencoder_dataset import AutoencoderDataset
from clearcut_research.pytorch.models.autoencoder.model import Autoencoder_Unet


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
    model = Autoencoder_Unet(encoder_name='resnet50')

    print("Loading model")
    model, device = UtilsFactory.prepare_model(model)

    train_df = pd.read_csv(args.train_df).to_dict('records')
    val_df = pd.read_csv(args.val_df).to_dict('records')
    ds = AutoencoderDataset(args.channels, args.dataset_path, args.image_size, args.batch_size, args.num_workers)
    loaders = ds.create_loaders(train_df, val_df)

    criterion = MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40], gamma=0.3)

    # model runner
    runner = SupervisedRunner()

    # model training
    runner.train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        callbacks=[
        ],
        scheduler=scheduler,
        loaders=loaders,
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
