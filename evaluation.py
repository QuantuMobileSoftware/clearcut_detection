import numpy as np
import argparse
import pandas as pd
import imageio
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


def compute_dice(true_positives, false_negatives, false_positives):
    return 1. * true_positives / (true_positives + false_negatives + false_positives)


def compute_iou(prediction, ground_truth):
    intersection = np.dot(prediction, ground_truth).sum()
    union = (prediction + ground_truth).sum() - intersection
    return intersection / union


def compute_metric(prediction, ground_truth, threshold):
    labels = np.unique(prediction)

    prediction_found = np.zeros(len(labels), dtype=bool)
    ground_truth_found = np.zeros(len(ground_truth), dtype=bool)

    for idx, label in tqdm(enumerate(labels)):
        if 0 < label < 255:
            prediction_instance = (prediction == label).astype(int)

            instance_ground_truth_iou = []
            for ground_truth_instance in ground_truth:
                iou = compute_iou(prediction_instance, ground_truth_instance)
                instance_ground_truth_iou.append(iou)
            max_iou = max(instance_ground_truth_iou)
            if max_iou > threshold:
                prediction_found[idx] = True
                ground_truth_found[np.argmax(instance_ground_truth_iou)] = True

    true_positives = prediction_found.sum()
    false_positives = np.logical_not(prediction_found).sum()
    false_negatives = np.logical_not(ground_truth_found).sum()
    return compute_dice(true_positives, false_negatives, false_positives)


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

    mark = markers.astype('uint8')
    mark = cv.bitwise_not(mark)

    return mark


def post_processing(prediction):
    return watershed_transformation(prediction)


def merge_adjacent_polygons(geoseries):
    return


def compute_average_metric(prediction, ground_truth):
    iou_at_thresholds = []
    for i in np.arange(0.5, 1, 0.05):
        iou_at_thresholds.append(compute_metric(prediction, ground_truth, i))
    return np.average(iou_at_thresholds)


def evaluate(ground_truth_path, predictions_path, pieces_info_path):
    pieces_info = pd.read_csv(pieces_info_path)

    metrics = []

    # for i in tqdm(range(len(pieces_info))):
    #     piece_name = pieces_info['piece_image'][i]
    #     extract from piece_name filename
    #     instances = []
    #     for instance_path in os.listdir(os.path.join(ground_truth_path, piece_name)):
    #         instances.append(imageio.imread(os.path.join(ground_truth_path, piece_name, instance_path)))

    piece_name = '20160103_66979721-be1b-4451-84e0-4a573236defd_rgb_13_21'

    instances = []
    for instance_path in os.listdir(os.path.join(ground_truth_path, piece_name)):
        if ".png" in instance_path and ".xml" not in instance_path:
            rgb_instance = cv.imread(os.path.join(ground_truth_path, piece_name, instance_path))
            bw_instance = cv.cvtColor(rgb_instance, cv.COLOR_BGR2GRAY)
            instances.append(bw_instance)

    prediction = cv.imread(os.path.join(predictions_path, f"{piece_name}.png"))
    post_processed_prediction = post_processing(prediction)
    metric = compute_average_metric(post_processed_prediction, instances)
    metrics.append(metric)

    return np.average(metrics)


if __name__ == "__main__":
    args = parse_args()
    print(evaluate(args.ground_truth_path, args.prediction_path, args.pieces_info_path))
