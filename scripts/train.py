import collections
import os
from shutil import move

import imageio
import numpy as np
import pandas as pd
import torch
from catalyst.dl.callbacks import DiceCallback, CheckpointCallback, InferCallback
from catalyst.dl.experiments import SupervisedRunner
from catalyst.dl.utils import UtilsFactory
from torch import cuda
from torch.backends import cudnn
from tqdm import tqdm

from data_split import add_record, get_image_info
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

        train_df = pd.read_csv(args.train_df)

        if os.path.isdir(args.unlabeled_data):
            for image_name in tqdm(os.listdir(args.unlabeled_data)):
                predicted_mask = image_predict(model, args.unlabeled_data, image_name, 320, channels_number=3)
                confidence = 1 - np.uint8(np.logical_and(0.3 < predicted_mask, predicted_mask < 0.7)).sum() / np.uint8(
                    predicted_mask > 0.5).sum()
                if confidence > 0.8:
                    print(f'Added {image_name} to train')
                    train_df = move_pseudo_labeled_to_train(image_name, predicted_mask, train_df)

        train_df.to_csv(args.train_df)


def move_pseudo_labeled_to_train(image_name, predicted_mask, train_df, img_size=320, mask_type="png", img_type="tiff"):
    pseudo_labeled_folder = "pseudo-labeled"
    pseudo_labeled_path = os.path.join(args.dataset_path, pseudo_labeled_folder)
    dataset_images_path = os.path.join(pseudo_labeled_path, 'images')
    dataset_masks_path = os.path.join(pseudo_labeled_path, 'masks')

    if not os.path.exists(pseudo_labeled_path):
        os.makedirs(pseudo_labeled_path, exist_ok=True)
        os.makedirs(dataset_images_path, exist_ok=True)
        os.makedirs(dataset_masks_path, exist_ok=True)
        print("Prediction directory created.")

    unlabeled_image_path = os.path.join(args.unlabeled_data, image_name)

    name, channel, position = get_image_info(image_name)

    move(unlabeled_image_path, os.path.join(dataset_images_path, image_name))
    imageio.imwrite(os.path.join(dataset_masks_path, name + '_' + channel + '_' + position + '.' + mask_type),
                    np.uint8(predicted_mask > 0.5) * 255)
    return add_record(train_df, pseudo_labeled_folder, image_name, channel, position, img_size, mask_type, img_type)


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
