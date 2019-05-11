import os

from image_division import divide_into_pieces
from binary_mask_converter import poly2mask, split_mask
from params import args


def preprocess(tiff_path, save_path, width, height, polys_path):
    for root, dirs, files in os.walk(tiff_path):
        for file in files:
            if file[-4:] != '.tif':
                continue
            tiff = os.path.join(root, file)
            image_path = os.path.join(save_path, file[:-4])
            divide_into_pieces(tiff, image_path, width, height)
            pieces_path = os.path.join(image_path, "masks")
            pieces_info = os.path.join(image_path, "image_pieces.csv")
            mask_path = poly2mask(polys_path, tiff, image_path)
            split_mask(mask_path, pieces_path, pieces_info)


if __name__ == '__main__':
    preprocess(
        args.tiff_path, args.save_path,
        args.width, args.height, args.polys_path)
