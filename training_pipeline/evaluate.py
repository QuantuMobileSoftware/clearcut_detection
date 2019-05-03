import os
from time import clock

import pandas as pd
import numpy as np
from keras.preprocessing.image import img_to_array
from PIL import Image
from tqdm import tqdm
import tensorflow as tf

from params import args

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def dice_coef(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def iou(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def calculate_metrics(base_mask, transformed_mask, eta=0.0000001):
    """
    Calculates IoU/Jaccard and F1/Dice metric from masked rasters
    :param base_mask: base image thresholded mask
    :param transformed_mask: transformed image thresholded mask
    :return: IoU/Jaccard and F1/Dice metric
    """
    base_mask = base_mask > 0
    transformed_mask = transformed_mask > 0
    inter = np.sum(base_mask * transformed_mask)
    union = np.sum(base_mask) + np.sum(transformed_mask)
    iou = inter / (union - inter + eta)
    fn = np.sum(base_mask) - inter
    fp = np.sum(transformed_mask) - inter
    dice = 2 * inter / (2 * inter + fp + fn + eta)
    return iou, dice

def evaluate(masks_dir, results_dir, tfr_df_name, r_type='nrg'):
    """
    Creates dataframe and tfrecords file for results visualization
    :param masks_dir: Ground truth masks dir
    :param results_dir: Predicted masks dir
    :param tfr_df_name: Name of output DataFrame and TFRecords, Will be saved to results_dir/<name>
    :param r_type: raster type, rgg or nrg
    :return:
    """
    test_df = pd.read_csv(args.test_df)
    thresh = args.threshold
    batch_size = 1
    nbr_test_samples = len(test_df)
    df = pd.DataFrame(columns=['name', 'score', 'img_width', 'img_height'])
    # img_sizes = []
    dices = []
    ious = []
    writer = tf.python_io.TFRecordWriter(os.path.join(os.path.dirname(results_dir), tfr_df_name.replace('.csv', '.tfrecords')))
    for i in tqdm(range(int(nbr_test_samples / batch_size))):
        try:
            masks = []
            results = []
            mask_filename = None
            img_size = None
            for j in range(batch_size):
                if i * batch_size + j < len(test_df):
                    row = test_df.iloc[i * batch_size + j]
                    mask_filename = row['name']
                    mask_path = os.path.join(masks_dir, row['folder'], 'nrg_masks', row['name'].replace(".JPG", ".png").replace(".jpg", ".png"))
                    img_path = os.path.join(masks_dir, row['folder'], r_type, row['name'].replace('nrg', r_type).replace('png', 'jpg'))
                    result_path = os.path.join(results_dir, row['name'].replace('nrg', 'nrg').replace('jpg', 'png'))
                    mask = Image.open(mask_path)
                    image = Image.open(img_path)
                    result = Image.open(result_path)
                    img_size = mask.size
                    masks.append(img_to_array(mask) > 128)
                    results.append(img_to_array(result) > 128)

            masks = np.array(masks)
            results = np.array(results)
            batch_dice = dice_coef(masks, results)
            batch_iou = iou(masks, results)
            dices.append(batch_dice)
            ious.append(batch_iou)
            df.loc[i] = [mask_filename, batch_dice, img_size[0], img_size[1]]
            
            # if row['name'].split('/')[-1].split('_')[1] == 'nrg':
            #     grove = row['name'].split('/')[-1].split('_')[0]
            # else:
            #     grove = row['name'].split('/')[-1].split('_')[0] + '_' + row['name'].split('/')[-1].split('_')[1]
            grove = row['grove']
            # if row['name'].split('/')[-1].split('_')[1] == 'nrg':
            #     grove = row['name'].split('/')[-1].split('_')[0]
            # else:
            #     grove = row['name'].split('/')[-1].split('_')[0] + '_' + row['name'].split('/')[-1].split('_')[1]
            img_raw = image.tobytes()
            msk_raw = mask.tobytes()
            rst_raw = result.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "img_height": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(img_size[1])])),
                "img_width": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(img_size[0])])),
                "dice_score": tf.train.Feature(float_list=tf.train.FloatList(value=[batch_dice])),
                "grove": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(grove)])),
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                "mask_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[msk_raw])),
                "result_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[rst_raw])),
                "img_name": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(img_path.split('/')[-1])])),
                "msk_name": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(mask_path.split('/')[-1])])),
                "rst_name": tf.train.Feature(
                    bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(result_path.split('/')[-1])])),
            }))
            writer.write(example.SerializeToString())

        except:
            print('wrong')
            continue
    print(np.mean(dices))
    writer.close()
    df.to_csv(os.path.join(os.path.dirname(results_dir), tfr_df_name), index=False)
    
if __name__ == '__main__':
    prediction_dir = args.pred_mask_dir
    mask_dir = args.test_mask_dir
    output_csv_name = args.output_csv
    evaluate(mask_dir,
             prediction_dir,
             output_csv_name,
             args.r_type)
    # evaluate('/media/user/5674E720138ECEDF/geo_data/manual_labelling/images_for_labeling',
    #          '/home/user/projects/geo/dl/unet/data/test_weights/output_rgg_cleaned_20_09',
    #          'df_tr_rgg_cleaned_20_09.csv')
    # evaluate('/home/user/projects/geo/geoslicer/output',
    #          '/home/user/projects/geo/dl/unet/data/test_weights/output_brazil',
    #          'df_tr_full_brazil.csv')