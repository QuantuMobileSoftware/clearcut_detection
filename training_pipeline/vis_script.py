import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import argparse


def visualize(image, mask, result):
    img_list = os.listdir(image)
    for image_name in img_list:
        img_path = os.path.join(image, image_name)
        mask_path = os.path.join(mask, image_name.replace('.jpg', '.png'))
        res_path = os.path.join(result, image_name.replace('.jpg', '.png'))
        print(img_path)
        img = Image.open(img_path)
        msk = Image.open(mask_path)
        rst = Image.open(res_path)
        fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
        ax = axes.ravel()
        print(np.array(img).shape)
        ax[0].imshow(np.array(img))
        ax[0].set_title('Image')
        ax[1].imshow(np.array(msk), cmap=plt.cm.gray, interpolation='nearest')
        ax[1].set_title('Mask')
        ax[2].imshow(np.array(rst), cmap=plt.cm.gray, interpolation='nearest')
        ax[2].set_title('Result')

        for a in ax:
            a.set_axis_off()

        fig.tight_layout()
        plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--img', type=str, default='data/out_data/test')
    arg('--mask', type=str, default='data/out_data/test_masks')
    arg('--result', '-r', type=str, default='data/out_data/output')
    args = parser.parse_args()

    visualize(args.img, args.mask, args.result)