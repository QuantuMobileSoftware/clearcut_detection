import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument

# arg('--loss_function', default='bce_dice')
# arg('--learning_rate', type=float, default=0.0001)

arg('--batch_size', type=int, default=8)
arg('--num_workers', type=int, default=4)
arg('--epochs', type=int, default=30)

arg('--logdir', default='../logs')
arg('--train_df', default='../../preprocessed_2016/train.csv')
arg('--val_df', default='../../preprocessed_2016/test.csv')
arg('--dataset_path', default='../../preprocessed_2016/')

arg('--img_width', type=int, default=224)
arg('--img_height', type=int, default=224)

arg('--network', default='linknet')

args = parser.parse_args()
