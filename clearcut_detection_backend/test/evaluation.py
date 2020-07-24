import os
import json
import yaml

import numpy as np
import pandas as pd
import rasterio
import geopandas

from tqdm import tqdm
from rasterio import features

from test.polygon_metrics import f1_score_evaluation, polygonize
from test.utils import GOLD_DICE, GOLD_F1SCORE, GOLD_IOU, SUCCESS_THRESHOLD, IOU_THRESHOLD
from test.test_data_prepare import get_gt_polygons

def dice_coef(y_true, y_pred, eps=1e-7):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + eps) / (np.sum(y_true_f) + np.sum(y_pred_f) + eps)


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


def get_raster_masks(reference_tif, model_result):
    raster = {}
    with rasterio.open(reference_tif) as src:
        filenames = {}
        filenames['mask'] = get_gt_polygons()
        filenames['predicted'] = os.path.join('data', model_result[0].get('polygons'))
        for name in filenames:
            gt_polygons = geopandas.read_file(filenames[name])
            gt_polygons = gt_polygons.to_crs(src.crs)
            raster[name] = features.rasterize(shapes=gt_polygons['geometry'],
                                              out_shape=(src.height, src.width),
                                              transform=src.transform,
                                              default_value=1)
        
    return raster

def load_config():
    with open('./model/predict_config.yml', 'r') as config:
        cfg = yaml.load(config, Loader=yaml.SafeLoader)

    models = cfg['models']
    save_path = cfg['prediction']['save_path']
    threshold = cfg['prediction']['threshold']
    input_size = cfg['prediction']['input_size']

    return models, save_path, threshold, input_size


def evaluate(model_result, test_tile_path):
    raster = get_raster_masks(test_tile_path['current'], model_result)
    _, _, _, size = load_config()
    
    res_cols = ['name', 'dice_score', 'iou_score', 'pixel_amount']
    test_df_results = pd.DataFrame(columns=res_cols)
    dices, ious = [], []
    test_polys, truth_polys = [], []
    for i in tqdm(range(raster['mask'].shape[0] // size)):
        for j in range(raster['mask'].shape[1] // size):
            instance_name = f'{i}_{j}'
            mask = raster['mask'][i*size : (i+1)*size, j*size : (j+1)*size]
            if mask.sum() > 0:
                prediction = raster['predicted'][i*size : (i+1)*size, j*size : (j+1)*size]
                test_polys.append(polygonize(prediction.astype(np.uint8)))
                truth_polys.append(polygonize(mask.astype(np.uint8)))

                dice_score = dice_coef(mask, prediction)
                iou_score = iou(mask, prediction, smooth=1.0)

                dices.append(dice_score)
                ious.append(iou_score)

                pixel_amount = mask.sum()

                test_df_results = test_df_results.append({'name': instance_name,
                                        'dice_score': dice_score, 'iou_score': iou_score, 'pixel_amount': pixel_amount}, ignore_index=True)

    log = pd.DataFrame(columns=['f1_score', 'threshold', 'TP', 'FP', 'FN'])
    for threshold in np.arange(0.1, 1, 0.1):
        F1score, true_pos_count, false_pos_count, false_neg_count, total_count = f1_score_evaluation(test_polys, truth_polys, threshold=threshold)
        log = log.append({'f1_score': round(F1score,4),
                          'threshold': round(threshold,2),
                          'TP':int(true_pos_count),
                          'FP':int(false_pos_count),
                          'FN':int(false_neg_count)}, ignore_index=True)
    
    return log, np.average(dices), np.average(ious)



def model_evaluate(model_result, test_tile_path):
    f1_score_test, dice, iou = evaluate(model_result, test_tile_path)

    f1_score_test = f1_score_test[f1_score_test['threshold'] == IOU_THRESHOLD]['f1_score'].to_numpy()
    f1_score_standard = GOLD_F1SCORE

    result = {}
    result['f1_score'] = float(f1_score_standard - f1_score_test[0])
    result['dice_score'] = GOLD_DICE - dice
    result['iou_score'] = GOLD_IOU - iou
    result['status'] = (result['f1_score'] < SUCCESS_THRESHOLD) \
                     & (result['dice_score'] < SUCCESS_THRESHOLD) \
                     & (result['iou_score'] < SUCCESS_THRESHOLD)

    if result['status']:
        result['status'] = str(result['status']).replace('True', 'success')
    else:
        result['status'] = str(result['status']).replace('False', 'failed')

    with open('test_status.json', 'w') as outfile:
        json.dump(result, outfile)
