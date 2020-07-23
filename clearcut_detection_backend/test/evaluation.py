import os
import json
import argparse

import imageio
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from PIL import Image
from tqdm import tqdm
from settings import DATA_DIR
from sklearn.metrics import f1_score, precision_score, recall_score, auc, precision_recall_curve

from polyeval import *
from utils import GOLD_STANDARD_F1SCORES, GOLD_DICE, GOLD_IOU, SUCCESS_THRESHOLD, IOU_THRESHOLD

def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for evaluating performance of the model.')
    parser.add_argument(
        '--datasets_path', '-dp', dest='datasets_path',
        default=f'{DATA_DIR}/predicted/masks',
        help='Path to the directory all the data')
    parser.add_argument(
        '--predictions_path', '-pp', dest='predictions_path',
        default=f'{DATA_DIR}/predicted/preds',
        help='Path to the directory with predictions')
    parser.add_argument(
        '--output_name', '-on', dest='output_name',
        default='inference', help='Name for output file')
    parser.add_argument(
        '--masks_folder', '-mf', dest='masks_folder',
        default='masks',
        help='Name of folder where masks are storing'
    )
    parser.add_argument(
        '--mask_type', '-mt', dest='mask_type',
        default='png',
        help='Type of mask file'
    )

    return parser.parse_args()

def dice_coef(y_true, y_pred, eps=1e-7):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + eps) / (np.sum(y_true_f) + np.sum(y_pred_f)+eps)


def iou(y_true, y_pred, smooth=1.0):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (1. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) - intersection + smooth)

def confusion_matrix(y_true, y_pred):
    mm, mn, nm, nn = 0, 0, 0, 0
    M, N = 0, 0
    for i in range(len(y_true)):
        if(y_true.iloc[i] == y_pred.iloc[i]):
            if(y_true.iloc[i] == 1):
                M += 1
                mm += 1
            else:
                N += 1
                nn += 1
        else:
            if(y_true.iloc[i] == 1):
                M += 1
                mn += 1
            else:
                N += 1
                nm += 1
    return mm, mn, nm, nn, M, N

def evaluate(datasets_path, predictions_path, output_name, masks_folder, mask_type):
    threshold = 0.3
    res_cols = ['name', 'dice_score', 'iou_score', 'pixel_amount']
    test_df_results = pd.DataFrame(columns=res_cols)
    dices, ious = [], []
    filenames = os.listdir(datasets_path)

    test_polys, truth_polys = [], []
    for instance_name in tqdm(filenames):
        prediction = cv.imread(os.path.join(predictions_path, instance_name))
        mask = cv.imread(os.path.join(datasets_path, instance_name))

        test_polys.append(polygonize(prediction[:,:,0].astype(np.uint8)))
        truth_polys.append(polygonize(mask[:,:,0].astype(np.uint8)))

        dice_score = dice_coef(mask / 255, (prediction / 255) > threshold)
        iou_score = iou(mask / 255, (prediction / 255) > threshold, smooth=1.0)

        dices.append(dice_score)
        ious.append(iou_score)

        pixel_amount = mask.sum() / 255

        test_df_results = test_df_results.append({'name': instance_name,
                                'dice_score': dice_score, 'iou_score': iou_score, 'pixel_amount': pixel_amount}, ignore_index=True)

    print("Average dice score - {0}".format(round(np.average(dices), 4)))
    print("Average iou  score - {0}".format(round(np.average(ious), 4)))

    log_save = os.path.join(predictions_path, f'{output_name}_f1score.csv')
    print(log_save)
    log = pd.DataFrame(columns=['f1_score', 'threshold', 'TP', 'FP', 'FN'])
    for threshold in np.arange(0.1, 1, 0.1):
        F1score, true_pos_count, false_pos_count, false_neg_count, total_count = evalfunction(test_polys, truth_polys, threshold=threshold)
        log = log.append({'f1_score': round(F1score,4),
                          'threshold': round(threshold,2),
                          'TP':int(true_pos_count),
                          'FP':int(false_pos_count),
                          'FN':int(false_neg_count)}, ignore_index=True)
    
    print(log)
    log.to_csv(log_save, index=False)

    test_df_results_path = os.path.join(predictions_path, f'{output_name}_dice.csv')
    test_df_results.to_csv(test_df_results_path, index=False)
    return log, np.average(dices), np.average(ious)



if __name__ == "__main__":
    args = parse_args()
    f1_score_test, dice, iou = evaluate(args.datasets_path, args.predictions_path, args.output_name, args.masks_folder, args.mask_type)
    f1_score_standard = pd.read_csv(GOLD_STANDARD_F1SCORES)

    f1_score_test = f1_score_test[f1_score_test['threshold'] <= IOU_THRESHOLD]
    f1_score_standard = f1_score_standard[f1_score_standard['threshold'] <= IOU_THRESHOLD]

    result = {}
    result['f1_score'] = np.mean(f1_score_standard['f1_score'].to_numpy() - f1_score_test['f1_score'].to_numpy())
    result['dice_score'] = GOLD_DICE - dice
    result['iou_score'] = GOLD_IOU - iou
    result['status'] = (result['f1_score'] < SUCCESS_THRESHOLD) \
                     & (result['dice_score'] < SUCCESS_THRESHOLD) \
                     & (result['iou_score'] < SUCCESS_THRESHOLD)

    if result['status']:
        result['status'] = str(result['status']).replace('True', 'success')
    else:
        result['status'] = str(result['status']).replace('False', 'failed')

    print(result)    
    with open('test_status.json', 'w') as outfile:
        json.dump(result, outfile)