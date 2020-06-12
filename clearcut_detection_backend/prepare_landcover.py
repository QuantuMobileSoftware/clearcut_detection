import os
import sys
import argparse
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling

def transform_crs(data_path, save_path, dst_crs = 'epsg:32636'):
    with rasterio.open(data_path) as src:
        transform, width, height = calculate_default_transform(src.crs,
                                                               dst_crs,
                                                               src.width,
                                                               src.height,
                                                               *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({'crs': dst_crs,
                       'transform': transform,
                       'width': width,
                       'height': height})
        with rasterio.open(save_path, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                      source=rasterio.band(src, i),
                      destination=rasterio.band(dst, i),
                      src_transform=src.transform,
                      src_crs=src.crs,
                      dst_transform=transform,
                      dst_crs=dst_crs,
                      resampling=Resampling.nearest)
        dst.close()
    src.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Script for preparing land cover map.'
    )
    parser.add_argument(
        '--data_path', '-tp', dest='data_path',
        required=True, help='Path to directory with land cover maps'
    )
    parser.add_argument(
        '--save_path', '-sp', dest='save_path',
        default='../data/auxiliary',
        help='Path to directory where data will be stored'
    )
    parser.add_argument(
      '--dst_crs', '-dc', dest='dst_crs',
      default='epsg:32636',
      help='Target coordinate system (default: epsg:32636)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if os.path.isfile(args.data_path):
        transform_crs(args.data_path, args.save_path, args.dst_crs)
    else:
        print(f"wrong data_path {args.data_path}")
        sys.exit()
