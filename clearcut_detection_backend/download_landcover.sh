# !/bin/bash
PATH_LANDCOVER=$1

if [ ! -f "$PATH_LANDCOVER/forest.tiff" ]; then
	apt update && apt-get install -y wget
	wget https://s3-eu-west-1.amazonaws.com/vito.landcover.global/2015/E020N60_ProbaV_LC100_epoch2015_global_v2.0.2_products_EPSG-4326.zip -O landcover.zip

	cp landcover.zip ./data/
	cd data && unzip landcover.zip

	readarray -d / -t strarr <<< "$PATH_LANDCOVER"
	RELATIVE_PATH="${strarr[-1]}"
	RELATIVE_PATH=$(echo $RELATIVE_PATH|tr -d '\n')
	mkdir $RELATIVE_PATH

	cp E020N60_ProbaV_LC100_epoch2015_global_v2.0.2_forest-type-layer_EPSG-4326.tif $RELATIVE_PATH/forest.tiff
	rm *.tif landcover.zip
fi
