import argparse
import os
import sys

import tensorflow as tf
import pandas as pd
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from tqdm import tqdm

class RecordsManager:
    """Class, that performs reading, writing and visualization of tfrecords files,
    that store cut rasters and their labels."""
    def __init__(self):
        self.tf_rec_files = []

    def write_to_tf_records(self, folders, tf_rec_names):
        """
        Write images from folders to tfrecords.
        :param folders: list of folders with images
        :param tf_rec_names:
        :return: -
        """
        classes = ['0/', '1/']
        cwd = os.getcwd()
        if type(folders) == str:
            folders = list(folders)
        for folder, tf_rec_name in zip(folders, tf_rec_names):
            writer = tf.python_io.TFRecordWriter(tf_rec_name)
            for index, name in enumerate(classes):
                class_path = os.path.join(cwd, folder, name)
                for img_name in os.listdir(class_path):
                    img_path = class_path + img_name
                    if os.path.isfile(img_path):
                        img = Image.open(img_path)
                        img = img.resize((100, 100))
                        img_raw = img.tobytes()
                        example = tf.train.Example(features=tf.train.Features(feature={
                            "label_true": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(name[0])])),
                            "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                            "img_name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(img_name)])),
                        }))
                        writer.write(example.SerializeToString())
            self.tf_rec_files.append(tf_rec_name)
            writer.close()

    def write_to_tf_records_from_csv(self, images_dir, csv_file, tf_rec_name):
        """
        Write images to tfrecords.
        :param images_dir: list of folders with images
        :param csv_file: file, containing
        :param tf_rec_name:
        :return: -
        """
        # images_dir = '/media/user/5674E720138ECEDF/geo_data/manual_labelling/data_for_train/test_masks'
        images_info = pd.read_csv(csv_file)
        images_info['grove'] = [
            i.split('/')[-1].split('_')[0] if i.split('/')[-1].split('_')[1] == 'nrg' else i.split('/')[-1].split('_')[
                                                                                               0] + '_' +
                                                                                           i.split('/')[-1].split('_')[
                                                                                               1] for i in images_info['name']]
        writer = tf.python_io.TFRecordWriter(tf_rec_name[0])
        for index, row in tqdm(images_info.iterrows()):
            res = row['name'].replace('nrg', 'rgg')
            # image_path = os.path.join(images_dir[0], row['path'])
            # mask_path = os.path.join(images_dir, row['name'])
            # result_path = os.path.join(images_dir, res).replace('full_test_masks', 'full_labelled').replace('.jpg', '.png')
            # img_path = os.path.join(images_dir, res.replace('nrg', 'rgg')).replace('.png', '.jpg').replace('full_test_masks', 'full_test')
            mask_path = row['name'].replace('output_fnt', 'test_masks')
            result_path = res.replace('test_masks', 'output_fnt').replace('.jpg', '.png').replace('rgg', 'rgg')
            img_path = res.replace('rgg', 'rgg').replace('.png', '.jpg').replace('output_fnt', 'test')
            img = Image.open(img_path)
            msk = Image.open(mask_path)
            rst = Image.open(result_path)
            # img = img.resize((100, 100))
            img_raw = img.tobytes()
            msk_raw = msk.tobytes()
            rst_raw = rst.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                "img_height": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row['img_height'])])),
                "img_width": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(row['img_width'])])),
                "dice_score": tf.train.Feature(float_list=tf.train.FloatList(value=[row['score']])),
                "grove": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(row['grove'])])),
                "img_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
                "mask_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[msk_raw])),
                "result_raw": tf.train.Feature(bytes_list=tf.train.BytesList(value=[rst_raw])),
                "img_name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(img_path.split('/')[-1])])),
                "msk_name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(mask_path.split('/')[-1])])),
                "rst_name": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(result_path.split('/')[-1])])),
            }))
            writer.write(example.SerializeToString())
        self.tf_rec_files.append(tf_rec_name)
        writer.close()

    def read_from_tf_records(self, filenames, mode=None):
        """
        Reads images, labels from tfrecords files.
        :param mode: set to 'pred', if there are predicted labels in tfrecords file
        :param filenames: list of tfrecord file names
        :return: lists of
        """
        images, image_names, labels_true, labels_pred, confidence = [], [], [], [], []
        if not filenames:
            filenames = self.tf_rec_files
        if type(filenames) == str:
            filenames = list(filenames)

        for filename in filenames:
            assert os.path.isfile(filename)
            for serialized_example in tf.python_io.tf_record_iterator(filename):
                example = tf.train.Example()
                example.ParseFromString(serialized_example)
                img_raw = example.features.feature['img_raw'].bytes_list.value
                label_true = example.features.feature['label_true'].int64_list.value
                image_name = example.features.feature['img_name'].bytes_list.value
                image = Image.frombytes('RGB', (100, 100), img_raw[0])
                images.append(image)
                labels_true.append(label_true)
                image_names.append(image_name)
                if mode == 'pred':
                    labels_pred.append(example.features.feature['label_pred'].int64_list.value)
                    confidence.append(example.features.feature['confidence'].float_list.value)

        return images, image_names, labels_true, labels_pred, confidence

    def plot_images(self, start_from, finish_on, filenames, mode, plot_size=4):
        """
        Visualise images and their labels using matplotlib.
        :param start_from: index to start visualization from, int
        :param finish_on: index to finish visualization on, int
        :param filenames: list of tfrecords files
        :param plot_size: number of images in row/columns to show, int
        :return: -
        """
        images, image_names, labels_true, labels_pred, confidence = self.read_from_tf_records(filenames=filenames, mode=mode)
        assert len(images) == len(labels_true)
        if not finish_on:
            finish_on = len(images)
        if labels_pred:
            hspace = 1.2
        else:
            hspace = 0.6

        i = start_from
        for _ in range(len(labels_true) // (plot_size ** 2)):
            fig, axes = plt.subplots(plot_size, plot_size)

            fig.subplots_adjust(hspace=hspace, wspace=0.3)
            for ax in axes.flat:
                i += 1
                if i < len(images):
                    ax.imshow(images[i], interpolation='spline16')
                    cls_true_name = labels_true[i]
                    image_name = '_'.join(str(image_names[i][0], 'utf-8').split('_'))[:-4]
                    if labels_pred:
                        cls_pred_name = labels_pred[i]
                        conf = confidence[i]
                        # print(conf)
                        xlabel = "{0}\nTrue_label: {1}\nPred_label: {2}\nConfidence: {3:.4f}".format(image_name,
                                                                                                 cls_true_name[0],
                                                                                                 cls_pred_name[0],
                                                                                                 conf[0])
                        if cls_pred_name[0] == 1:
                            plt.setp(ax.spines.values(), color='green')
                            plt.setp(ax.spines.values(), linewidth=3)
                        else:
                            plt.setp(ax.spines.values(), color='red')
                            plt.setp(ax.spines.values(), linewidth=3)
                    else:
                        xlabel = "{0}\nTrue_label: {1}".format(image_name,
                                                               cls_true_name[0])
                    ax.set_xlabel(xlabel)

                ax.set_xticks([])
                ax.set_yticks([])

            # Try to make the window full-screen
            try:
                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
            except:
                pass
            plt.show()
            if i >= finish_on:
                break

    @staticmethod
    def show_tfrecords_gcp(filename):
        """Service method, to be removed"""
        images, ids_ext, ids_int = [], [], []
        for serialized_example in tf.python_io.tf_record_iterator(filename):
            example = tf.train.Example()
            example.ParseFromString(serialized_example)
            img_raw = example.features.feature['img_raw'].bytes_list.value
            id_ext = example.features.feature['image_id_external'].int64_list.value
            id_int = example.features.feature['image_id_internal'].bytes_list.value
            image = Image.frombytes('RGB', (100, 100), img_raw[0])
            images.append(image)
            ids_ext.append(id_ext)
            ids_int.append(id_int)

        hspace = 0.9
        # print(len(images))
        i = 0
        for _ in range(len(images) // (50 ** 2)):
            fig, axes = plt.subplots(50, 50)

            fig.subplots_adjust(hspace=hspace, wspace=0.3)
            for ax in axes.flat:
                i += 1
                if i < len(images):
                    ax.imshow(images[i], interpolation='spline16')
                    xlabel = "IDint: {0}\nIDext: {1}".format(ids_int[i],
                                                           ids_ext[i])
                    ax.set_xlabel(xlabel)

                ax.set_xticks([])
                ax.set_yticks([])

            # Try to make the window full-screen
            try:
                mng = plt.get_current_fig_manager()
                mng.resize(*mng.window.maxsize())
            except:
                pass
            plt.show()


# if __name__ == "__main__":
#     rec_man = RecordsManager()
#     rec_man.show_tfrecords_gcp("/home/pcs/PycharmProjects/classification/image_indices.tfrecords")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--writer', action='store_true', help='Run tfrecords writer; Need to specify images_dir.')
    parser.add_argument('--images_dir', nargs='+',
                        help='Specify directory with class folders, that needs to be written to tfrecords files.')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='Visualize images and labels from tfrecords files; Need to specify the file(s) path(s).')
    parser.add_argument('--paths_to_tfrecords', nargs='+',
                        help='List of paths to tfrecords files, used in visualization.')
    parser.add_argument('--start_from', default=0, help='Index of first image to inspect.', type=int)
    parser.add_argument('--finish_on', help='Index of the last image to inspect.', type=int)
    parser.add_argument('--mode', type=str, default=None,
                        help='Write with predicted labels, or only true. If predicted - set to pred')
    parser.add_argument('--csv_file', type=str, help='File with image dirs and true and predicted labels.')

    args = parser.parse_args()
    data_manager = RecordsManager()
    if args.writer and args.mode == 'pred':
        if not os.path.isfile(args.csv_file):
            print("csv file does not exist")
            sys.exit(1)
        data_manager.write_to_tf_records_from_csv(images_dir=args.images_dir,
                                                  csv_file=args.csv_file,
                                                  tf_rec_name=args.paths_to_tfrecords)
    elif args.writer:
        data_manager.write_to_tf_records(folders=args.images_dir,
                                         tf_rec_names=args.paths_to_tfrecords)
    if args.visualize:
        data_manager.plot_images(start_from=args.start_from, finish_on=args.finish_on,
                                 filenames=args.paths_to_tfrecords, mode=args.mode)

