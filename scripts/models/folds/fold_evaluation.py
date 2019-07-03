import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for evaluating performance of the model.')
    parser.add_argument(
        '--datasets_path', '-dp', dest='datasets_path',
        required=True, help='Path to the directory all the data')
    parser.add_argument(
        '--prediction_path', '-pp', dest='prediction_path',
        required=True, help='Path to the directory with fold predictions')
    parser.add_argument(
        '--test_df_path', '-tp', dest='test_df_path',
        required=True, help='Path to the test dataframe with image names')
    parser.add_argument(
        '--output_name', '-on', dest='output_name', default='prediction_metrics',
        help='Name for output file')
    parser.add_argument(
        '--folds', '-f', dest='folds',
        required=True, help='Number of folds', type=int)
    return parser.parse_args()


def dice_coef(y_true, y_pred, eps=1e-7):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + eps) / (np.sum(y_true_f) + np.sum(y_pred_f))


def evaluate(datasets_path, predictions_path, test_df_path, output_name, folds, images_folder="images",
             image_type="tiff", masks_folder="masks", mask_type="png"):
    filenames = pd.read_csv(test_df_path)
    writer = tf.python_io.TFRecordWriter(
        os.path.join(os.path.dirname(predictions_path), output_name + '.tfrecords'))

    dices = []

    for ind, image_info in tqdm(filenames.iterrows()):

        name = image_info["name"] + '_' + image_info["position"]

        image = Image.open(os.path.join(datasets_path, image_info["dataset_folder"], images_folder,
                                        name + '.' + image_type))
        mask = Image.open(os.path.join(datasets_path, image_info["dataset_folder"], masks_folder,
                                       name + '.' + mask_type))
        img_size = image.size
        prediction = np.zeros(img_size, dtype=np.float)

        for fold_dir in os.listdir(predictions_path):
            if os.path.isdir(os.path.join(predictions_path, fold_dir)):
                prediction += np.array(
                    Image.open(os.path.join(predictions_path, fold_dir, 'predictions', name + ".png")))

        prediction = prediction / folds

        mask_array = np.array(mask)

        dice_score = dice_coef(mask_array / 255, (prediction / 255) > 0.5)
        dices.append(dice_score)

        prediction = prediction.astype(np.uint8)

        img_raw = image.tobytes()
        msk_raw = mask.tobytes()
        pred_raw = Image.fromarray(prediction, 'L').tobytes()

        example = tf.train.Example(features=tf.train.Features(feature={
            "img_height": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(img_size[1])])),
            "img_width": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(img_size[0])])),
            "dice_score": tf.train.Feature(float_list=tf.train.FloatList(value=[dice_score])),
            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
            "mask_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[msk_raw])),
            "pred_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[pred_raw])),
            "metric": tf.train.Feature(float_list=tf.train.FloatList(value=[0.])),
            "img_name": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(name)])),
            "msk_name": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(name)])),
            "pred_name": tf.train.Feature(
                bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(name)])),
        }))
        writer.write(example.SerializeToString())

    print("Average dice score - {0}".format(round(np.average(dices), 4)))


if __name__ == "__main__":
    args = parse_args()
    evaluate(args.datasets_path, args.prediction_path, args.test_df_path, args.output_name, args.folds)
