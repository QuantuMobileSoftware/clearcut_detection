# NeuralNet polygonization


## Setup environment

Install all the required packages:
```
pip install -r requirements.txt
```

## Run training with CSSE-Resnet50

```
python train.py\
--batch_size 32\
--epochs 200\
--preprocessing_function tf\
--learning_rate 0.00001\
--network csse_resnet50\
--loss_function boot_crossentropy\
--weights models/rgg/csse_resnet50--loss-bce_dice-fold_None-2240.000100-86-0.0970888-0.9244311-93.4183056.h5
```

## Run predict on test data
```
python predict_masks.py\
--weights models/rgg/csse_resnet50--loss-bce_dice-fold_None-2240.000100-86-0.0970888-0.9244311-93.4183056.h5\
--pred_mask_dir data/test_weights/output_csse_rgg_cleaned_20_09\
--network csse_resnet50
```

## Mask list of rasters
```
python mask_raster_unet.py \
--weights models/rgg/csse_resnet50--loss-bce_dice-fold_None-2240.000100-86-0.0970888-0.9244311-93.4183056.h5 \
--network csse_resnet50 \
--preprocessing_function tf \
--inp_list input_list
```

input_list - json file with pathes to the raster lists that will be used for RGG generation.
Raster list example, for RGB you need only rgb path:
```bash
{
"nir":"/media/user/5674E720138ECEDF/geo_data/data/Suntreat/s4/pix4d/pix4d_sq/res_TV_TV3/-L4AfQKCoDRlbRWjmxZa/project/project/4_index/reflectance/tiles/project_noalpha_reflectance_nir_1_1.tif",
"green":"/media/user/5674E720138ECEDF/geo_data/data/Suntreat/s4/pix4d/pix4d_sq/res_TV_TV3/-L4AfQKCoDRlbRWjmxZa/project/project/4_index/reflectance/tiles/project_noalpha_reflectance_green_1_1.tif",
"red": "/media/user/5674E720138ECEDF/geo_data/data/Suntreat/s4/pix4d/pix4d_sq/res_TV_TV3/-L4AfQKCoDRlbRWjmxZa/project/project/4_index/reflectance/tiles/project_noalpha_reflectance_red_1_1.tif"
}
```
## TF records generation from predicted masks

```
python evaluate.py \
    --pred_mask_dir /home/user/projects/geo/dl/unet/data/test_weights/output_rgg_cleaned_20_09 \
    --test_mask_dir /media/user/5674E720138ECEDF/geo_data/manual_labelling/images_for_labeling \
    --output_csv df_tr_rgg_cleaned_20_09.csv \
    --test_df /media/user/5674E720138ECEDF/geo_data/train_only_man.df
```

## Visualize data

Folder notebooks contains notebooks for visualization and preparing the data.

You can find visualization demo in TFR_visualizaer.ipynb
To visualize .csv dataframe run Data_analyzer.ipynb. Ther you can plot the score distribution.