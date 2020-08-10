# ClearCut: Planet Scripts

There are two executable scripts: 
 - **planet_searcher.py**
 - **planet_loader.py**
 
 ## planet_searcher: 
 
 Search quality **PSOrthoTile** images, using Planet API, download thumbnails and draw user AOI
 on previews images.
 
 #### Run planet_searcher.py:
 
 `python planet_searcher.py <path to credentials.json> <path to .geojson>`
 
 `python planet_searcher.py credentials.json input/Kochetok_forest.geojson --start 2020-07-19 --end 2020-07-19  --width 2048`

  
 To see all args, execute
 
 `python planet_searcher.py -h`
 
 ## planet_loader: 
 
Load selected images
 
 #### Run planet_loader.py:
 
 `python planet_loader.py <path to credentials.json file> <to .json file>`
 
 `python planet_loader.py credentials.json load_assets.json --item_type PSScene4Band`
 
 To see all args, execute
 
 `python planet_loader.py -h`
 