import argparse
import os
from shutil import move

import imageio
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from catalyst.dl.utils import UtilsFactory
from tqdm import tqdm

from clearcut_research.pytorch.dataset import add_record
from clearcut_research.pytorch.models.utils import get_model
from clearcut_research.pytorch.train import train
from clearcut_research.pytorch.utils import get_image_info, count_channels


def parse_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument

    arg('--batch_size', type=int, default=8)
    arg('--num_workers', type=int, default=4)
    arg('--epochs', '-e', type=int, default=100)
    arg('--pseudolabel_iter', '-pi', type=int, default=2)

    arg('--logdir', default='../logs')
    arg('--train_df', '-td', default='../data/train_df.csv')
    arg('--val_df', '-vd', default='../data/val_df.csv')
    arg('--dataset_path', '-dp', default='../data/input', help='Path to the data')
    arg('--unlabeled_data', '-ud', default='../data/unlabeled_data')

    arg('--image_size', '-is', type=int, default=224)

    arg('--network', '-n', default='unet50')

    arg(
        '--channels', '-ch',
        default=[
            'rgb', 'ndvi', 'ndvi_color',
            'b2', 'b3', 'b4', 'b8'
        ], nargs='+', help='Channels list')

    return parser.parse_args()


def load_model(network, model_weights_path):
    model = get_model(network)
    model, device = UtilsFactory.prepare_model(model)
    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def pseudo_labeling(args, eps=1e-7, confidence_threshold=0.8, prediction_threshold=0.3):
    train(args)

    for i in range(args.pseudolabel_iter):
        model = load_model(args.network, f"{args.logdir}/checkpoints/best.pth")
        train_df = pd.read_csv(args.train_df)
        added_to_train = 0
        if os.path.isdir(args.unlabeled_data):
            for image_name in tqdm(os.listdir(args.unlabeled_data)):
                predicted_mask = image_labeling(model, args.unlabeled_data, image_name, args.image_size,
                                                channels_number=count_channels(args.channels))
                confidence = 1 - np.uint8(np.logical_and(0.3 < predicted_mask, predicted_mask < 0.7)).sum() / (np.uint8(
                    predicted_mask > 0.5).sum() + eps)
                if confidence > confidence_threshold and \
                        predicted_mask[predicted_mask > prediction_threshold].sum() > 50:
                    print(f'Added {image_name} to train')
                    added_to_train += 1
                    train_df = move_pseudo_labeled_to_train(image_name, predicted_mask, train_df, prediction_threshold)
        print(f'Added {added_to_train} images to train')
        train_df.to_csv(args.train_df)

        train(args)


def image_labeling(model, unlabeled_data, image_name, img_size, channels_number=3):
    image_path = os.path.join(unlabeled_data, image_name)

    img = Image.open(image_path)

    img_tensor = transforms.ToTensor()(img)

    prediction = model.predict(img_tensor.view(1, channels_number, img_size, img_size).cuda())

    result = prediction.view(img_size, img_size).detach().cpu().numpy()
    return result


def move_pseudo_labeled_to_train(image_name, predicted_mask, train_df, prediction_threshold, mask_type="png",
                                 pseudo_labeled_folder="pseudo-labeled"):
    pseudo_labeled_path = os.path.join(args.dataset_path, pseudo_labeled_folder)
    dataset_images_path = os.path.join(pseudo_labeled_path, 'images')
    dataset_masks_path = os.path.join(pseudo_labeled_path, 'masks')

    if not os.path.exists(pseudo_labeled_path):
        os.makedirs(pseudo_labeled_path, exist_ok=True)
        os.makedirs(dataset_images_path, exist_ok=True)
        os.makedirs(dataset_masks_path, exist_ok=True)
        print("Prediction directory created.")

    unlabeled_image_path = os.path.join(args.unlabeled_data, image_name)

    name, position = get_image_info(image_name)

    move(unlabeled_image_path, os.path.join(dataset_images_path, image_name))
    imageio.imwrite(os.path.join(dataset_masks_path, name + '_' + position + '.' + mask_type),
                    np.uint8(predicted_mask > prediction_threshold) * 255)
    return add_record(train_df, pseudo_labeled_folder, name, position)


if __name__ == '__main__':
    args = parse_args()
    pseudo_labeling(args)
