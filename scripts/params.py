import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument

# arg('--loss_function', default='bce_dice')
# arg('--learning_rate', type=float, default=0.0001)

arg('--batch_size', type=int, default=8)
arg('--num_workers', type=int, default=4)
arg('--epochs', type=int, default=30)

arg('--logdir', default='../logs')
arg('--train_df',
    default='../../preprocessed_2016/20160103_66979721-be1b-4451-84e0-4a573236defd_rgb/'
            '20160103_66979721-be1b-4451-84e0-4a573236defd_rgb_train.csv')
arg('--val_df',
    default='../../preprocessed_2016/20160103_66979721-be1b-4451-84e0-4a573236defd_rgb/'
            '20160103_66979721-be1b-4451-84e0-4a573236defd_rgb_test.csv')
arg('--dataset_path', default='../../preprocessed_2016/20160103_66979721-be1b-4451-84e0-4a573236defd_rgb')

arg('--img_width', type=int, default=224)
arg('--img_height', type=int, default=224)

arg('--network', default='resnet34')

args = parser.parse_args()
