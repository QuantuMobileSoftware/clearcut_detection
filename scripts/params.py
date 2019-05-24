import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument

# arg('--ground_truth_path')
# arg('--prediction_path')
# arg('--pieces_info_path')
#
# arg('--models_dir', default='models')
# arg('--network', default='simple_unet')
# arg('--loss_function', default='bce_dice')
# arg('--input_width', type=int, default=224)
# arg('--learning_rate', type=float, default=0.0001)
# arg('--show_summary', type=bool, default=False)
arg('--batch_size', type=int, default=8)
arg('--num_workers', type=int, default=4)
arg('--epochs', type=int, default=30)

arg('--logdir', default='./logs')
arg('--train_df',
    default='../preprocessed_data/20160103_66979721-be1b-4451-84e0-4a573236defd_rgb/'
            '20160103_66979721-be1b-4451-84e0-4a573236defd_rgb_train.csv')
arg('--val_df',
    default='../preprocessed_data/20160103_66979721-be1b-4451-84e0-4a573236defd_rgb/'
            '20160103_66979721-be1b-4451-84e0-4a573236defd_rgb_test.csv')
arg('--dataset_path', default='../preprocessed_data/20160103_66979721-be1b-4451-84e0-4a573236defd_rgb/')

arg('--img_width', type=int, default=224)
arg('--img_height', type=int, default=224)
# arg('--clr', default="0.001,0.006,300.0,triangular")
# arg('--use_clr', default=False)

args = parser.parse_args()
