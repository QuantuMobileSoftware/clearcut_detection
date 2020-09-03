# Tables

## clearcuts_tile
Store information about tiles
* id - int: primary key
* tile_index - str: tile name
* is_tracked - int:  0 - system will not track this tile, 1 - system will track this tile
* first_date - date: first date information was collected for this tile
* last_date - date: last date information was collected for this tile


## clearcuts_zone
* id - int: primary key
* tile - int: foreign key to clearcuts_tile.id


## clearcuts_clearcut
Store information about polygons which are not clouds and which are forest
* id - int: primary key
* image_date_previous - date: date of early photo for prediction
* image_date_current - date: date of later photo for prediction
* area - float: area of polygon in square meters 
* forest- int: 0 - is not forest, 1 - is forest
* clouds - int: 0 - is not clouds, 1 - is cloud
* centroid - GEOMETRY POINT: 
* zone - int: foreign key to clearcuts_zone.id
* mpoly - GEOMETRY POLYGON: srid=4326, spatial index field, store polygons
* status - int: 0 - field is not verified, 1 - field is valid, -1 - field is not valid


## clearcuts_notclearcut
Store information about polygons which are clouds or which are not forest
* id - int: primary key
* image_date_previous - date: date of early photo for prediction
* image_date_current - date: date of later photo for prediction
* area - float: area of polygon in square meters 
* forest- int: 0 - is not forest, 1 - is forest
* clouds - int: 0 - is not clouds, 1 - is cloud
* centroid - GEOMETRY POINT: 
* zone - int: foreign key to clearcuts_zone.id
* mpoly - GEOMETRY POLYGON: srid=4326, store polygons
* status - int: 0 - field is not verified, 1 - field is valid, -1 - field is not valid


## clearcuts_run_update_task
Store information about update tasks
* tile -int: foreign key to clearcuts_tile.id
* path_type - int: 0 if local file system
* path_img_0 - str: path to early photo for prediction
* path_img_1 - str: path to later photo for prediction
* image_date_0 - date: date of early photo for prediction
* image_date_1 - date: date of later photo for prediction
* path_clouds_0 - str: path to early photo of clouds
* path_clouds_1 - str: path to later photo of clouds
* result - str: path to result geo.json file with prediction
* date_created - datetime: datetime when task was created 
* date_started - datetime: datetime when task was started
* date_finished - datetime: datetime when task was finished


## downloader_sourcejp2images
Store information about downloaded images
tile and image_date are unique together fields
* tile - int: foreign key to clearcuts_tile.id
* image_date -  date: date of image creation
* tile_uri - str: prefix to get images from google storage_bucket
* cloud_coverage - float: percent of image cloud coverage
* nodata_pixel - float: percent of image nodata pixel
* source_tci_location - str: path to tci image was downloaded
* source_b04_location - str: path to b04 image was downloaded
* source_b08_location - str: path to b08 image was downloaded
* source_b8a_location - str: path to b8a image was downloaded
* source_b11_location - str: path to b11 image was downloaded
* source_b12_location - str: path to b12 image was downloaded
* source_clouds_location - str: path to clouds image was downloaded
* is_downloaded - int: number of images were downloaded, 7 - all images, -1 - error when downloaded 
* is_new - int: 0 - this image was prepared already, 1 - this image is not prepared
* check_date - datetime: datetime when image was checked


## tiff_prepare_prepared
Store information about prepared images
tile and image_date are unique together fields
* tile - int: foreign key to clearcuts_tile.id
* image_date -  date: date of image creation
* model_tiff_location - str: path to prepared tiff image
* cloud_tiff_location - str: path to prepared tiff clouds image
* success - int: image preparation was succeeded
* is_new - int: 0 - image was predicted, 1 - image will be used at the next prediction
* prepare_date - datetime: datetime when image was prepared
