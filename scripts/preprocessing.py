import os

from image_division import divide_into_pieces
from binary_mask_converter import poly2mask, split_mask

def preprocess(tiff_path, save_path, width, height, polys_path):
    for root, dirs, files in os.walk(TIFF_PATH):
        for file in files:
            if file[-4:] != '.tif':
                continue
            tiff = '{}{}'.format(root, file)
            image_path = '{}/{}'.format(save_path, file[:-4])
            divide_into_pieces(tiff, image_path, width, height)
            pieces_path = '{}/masks'.format(image_path)
            pieces_info = '{}/image_pieces.csv'.format(image_path)
            mask_path = poly2mask(polys_path, tiff, image_path)
            split_mask(mask_path, pieces_path, pieces_info)

if __name__ == '__main__':
    preprocess(
        args.tiff_path, args.save_path,
        args.width, args.height, args.polys_path)