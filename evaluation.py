import numpy as np
import argparse
import pandas as pd
import imageio
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for evaluating performance of the model.')
    parser.add_argument(
        '--ground_truth_path', '-gtp', dest='ground_truth_path',
        required=True, help='Path to the directory with ground truth masks')
    parser.add_argument(
        '--prediction_path', '-pp', dest='prediction_path',
        required=True, help='Path to the directory with predictions')
    parser.add_argument(
        '--pieces_info_path', '-pip', dest='pieces_info_path',
        required=True, help='Path to the image pieces info')

    return parser.parse_args()


def compute_dice(prediction, ground_truth):
    prediction = np.asarray(prediction).astype(np.bool)
    ground_truth = np.asarray(ground_truth).astype(np.bool)

    true_positives = np.logical_and(prediction, ground_truth).sum()
    tp_fp = prediction.sum()
    tp_fn = ground_truth.sum()

    return 2. * true_positives / (tp_fp + tp_fn)


def compute_iou(prediction, ground_truth):
    prediction = np.asarray(prediction).astype(np.bool)
    ground_truth = np.asarray(ground_truth).astype(np.bool)
    intersection = np.logical_and(prediction, ground_truth).sum()
    union = np.logical_or(prediction, ground_truth).sum()
    return intersection / union


def watershed_transformation(prediction):
    return 1


def post_processing(prediction):
    return watershed_transformation(prediction)


def merge_adjacent_polygons(geoseries):



def evaluation(ground_truth_path, predictions_path, pieces_info_path):

    pieces_info = pd.read_csv(pieces_info_path)

    # r=root, d=directories, f = files
    for image in tqdm(os.listdir(ground_truth_path)):
        ground_truth = imageio.imread(f"{ground_truth_path}/{image}")
        prediction = imageio.imread(f"{predictions_path}/{image}")
        post_processed_prediction = post_processing(prediction)


if __name__ == "__main__":
    args = parse_args()
    evaluation(args.ground_truth_path, args.prediction_path, args.pieces_info_path)
