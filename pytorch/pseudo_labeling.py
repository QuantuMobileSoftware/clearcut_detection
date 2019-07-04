from train import train
from prediction import image_labeling
from models.utils import get_model

from tqdm import tqdm
from catalyst.dl.utils import UtilsFactory
from datasets import add_record
from utils import get_image_info
import pandas as pd
import os
import numpy as np
import torch
from shutil import move
import imageio

from params import args


def pseudo_labeling(eps=1e-7, confidence_threshold=0.8):
    train()

    for i in range(args.pseudolabel_iter):
        model = load_model(args.network, f"{args.logdir}/checkpoints/best.pth")
        train_df = pd.read_csv(args.train_df)
        added_to_train = 0
        if os.path.isdir(args.unlabeled_data):
            for image_name in tqdm(os.listdir(args.unlabeled_data)):
                predicted_mask = image_labeling(model, args.unlabeled_data, image_name, 320, channels_number=3)
                confidence = 1 - np.uint8(np.logical_and(0.3 < predicted_mask, predicted_mask < 0.7)).sum() / (np.uint8(
                    predicted_mask > 0.5).sum() + eps)
                if confidence > confidence_threshold and predicted_mask[predicted_mask > 0.3].sum() > 50:
                    print(f'Added {image_name} to train')
                    added_to_train += 1
                    train_df = move_pseudo_labeled_to_train(image_name, predicted_mask, train_df)
        print(f'Added {added_to_train} images to train')
        train_df.to_csv(args.train_df)

        train()


def load_model(network, model_weights_path):
    model = get_model(network)
    model, device = UtilsFactory.prepare_model(model)
    checkpoint = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def move_pseudo_labeled_to_train(image_name, predicted_mask, train_df, mask_type="png"):
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

    name, position = get_image_info(image_name)

    move(unlabeled_image_path, os.path.join(dataset_images_path, image_name))
    imageio.imwrite(os.path.join(dataset_masks_path, name + '_' + position + '.' + mask_type),
                    np.uint8(predicted_mask > 0.3) * 255)
    return add_record(train_df, pseudo_labeled_folder, name, position)


if __name__ == '__main__':
    pseudo_labeling()
