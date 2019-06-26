import collections
import os

import numpy as np
import torch
from catalyst.dl.callbacks import DiceCallback, CheckpointCallback, InferCallback
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.utils import UtilsFactory
from torch import cuda
from torch.backends import cudnn
from tqdm import tqdm

from datasets import create_loaders
from losses import BCE_Dice_Loss
from models.utils import get_model
from params import args
from prediction import image_predict


def set_random_seed(seed):
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    torch.manual_seed(seed)
    if cuda.is_available():
        cuda.manual_seed_all(seed)

    print('Random seed:', seed)


def main():
    pseudo_labeling()


def pseudo_labeling():
    for i in range(args.pseudolabel_iter):
        train()

        model = get_model(args.network)
        model, device = UtilsFactory.prepare_model(model)
        checkpoint = torch.load(f"{args.logdir}/checkpoints/best.pth", map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])

        for image_name in tqdm(os.listdir(args.raw_data)):
            image_predict(model, args.unlabeled_data, image_name, args.dataset_path, 320, channels_number=3)


def train():
    set_random_seed(42)

    model = get_model(args.network)

    print("Loading model")
    model, device = UtilsFactory.prepare_model(model)

    loaders = create_loaders(args.train_df, args.val_df)

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
    checkpoint(model, runner, loaders)


def checkpoint(model, runner, loaders):
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
