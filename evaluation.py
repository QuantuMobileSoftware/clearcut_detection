import numpy as np
import argparse
import pandas as pd
from tqdm import tqdm
import os
import cv2 as cv


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


def watershed_transformation(prediction):
    kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)

    imgLaplacian = cv.filter2D(prediction, cv.CV_32F, kernel)
    sharp = np.float32(prediction)
    imgResult = sharp - imgLaplacian

    imgResult = np.clip(imgResult, 0, 255)
    imgResult = imgResult.astype('uint8')

    bw = cv.cvtColor(prediction, cv.COLOR_BGR2GRAY)
    _, bw = cv.threshold(bw, 40, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)

    dist = cv.distanceTransform(bw, cv.DIST_L2, 3)
    cv.normalize(dist, dist, 0, 1.0, cv.NORM_MINMAX)
    _, thresholded = cv.threshold(dist, 0.2, 1.0, cv.THRESH_BINARY)

    kernel1 = np.ones((3, 3), dtype=np.uint8)
    dilated = cv.dilate(thresholded, kernel1)

    dist_8u = dilated.astype('uint8')
    contours, _ = cv.findContours(dist_8u, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    markers = np.zeros(dilated.shape, dtype=np.int32)

    for i in range(len(contours)):
        cv.drawContours(markers, contours, i, (i + 1), -1)

    cv.circle(markers, (5, 5), 3, (255, 255, 255), -1)
    cv.watershed(imgResult, markers)

    return markers


def post_processing(prediction):
    return watershed_transformation(prediction)


def dice_coef(true_positives, false_positives, false_negatives):
    if true_positives + false_negatives + false_positives == 0:
        return 1
    return (2. * true_positives) / (2. * true_positives + false_positives + false_negatives)


def iou(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def compute_iou_matrix(markers, instances):
    labels = np.unique(markers)

    labels = labels[labels < 255]
    labels = labels[labels > 0]

    iou_matrix = np.zeros((len(labels), len(instances)), dtype=np.float32)

    for i, label in enumerate(labels):
        prediction_instance = (markers == label).astype(int)

        for j, ground_truth_instance in enumerate(instances):
            iou_value = iou(prediction_instance, ground_truth_instance)
            iou_matrix[i, j] = iou_value

    return iou_matrix


def compute_metric_at_thresholds(iou_matrix):
    dices = []
    if iou_matrix.shape == (0, 0):
        return 1
    for threshold in np.arange(0.5, 1, 0.05):
        true_positives = (iou_matrix.max(axis=1) > threshold).sum()
        false_positives = (iou_matrix.max(axis=1) <= threshold).sum()
        false_negatives = (iou_matrix.max(axis=0) <= threshold).sum()
        dices.append(dice_coef(true_positives, false_positives, false_negatives))
    return np.average(dices)


def evaluate(ground_truth_path, predictions_path, pieces_info_path):
    pieces_info = pd.read_csv(pieces_info_path)

    metrics = []

    for i in tqdm(range(len(pieces_info))):
        piece_name = pieces_info['piece_image'][i]
        filename, file_extension = os.path.splitext(piece_name)

        prediction = cv.imread(os.path.join(predictions_path, filename) + ".png")
        instances = []

        for instance_path in os.listdir(os.path.join(ground_truth_path, filename)):
            if ".png" in instance_path and ".xml" not in instance_path:
                rgb_instance = cv.imread(os.path.join(ground_truth_path, filename, instance_path))
                bw_instance = cv.cvtColor(rgb_instance, cv.COLOR_BGR2GRAY)
                instances.append(bw_instance)

        markers = post_processing(prediction)
        iou_matrix = compute_iou_matrix(markers, instances)
        metric = compute_metric_at_thresholds(iou_matrix)
        metrics.append(metric)

    return np.average(metrics)


if __name__ == "__main__":
    args = parse_args()
    print(f"Metrics value - {round(evaluate(args.ground_truth_path, args.prediction_path, args.pieces_info_path), 4)}")
