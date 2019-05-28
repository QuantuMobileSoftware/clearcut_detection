import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument

# arg('--loss_function', default='bce_dice')
# arg('--learning_rate', type=float, default=0.0001)

arg('--batch_size', type=int, default=4)
arg('--num_workers', type=int, default=4)
arg('--epochs', type=int, default=30)

arg('--logdir', default='../logs')
arg('--train_df', default='../data/train_df.csv')
arg('--val_df', default='../data/test_df.csv')
arg('--dataset_path', default='../data')

arg('--img_width', type=int, default=224)
arg('--img_height', type=int, default=224)

arg('--network', default='resnet50')

args = parser.parse_args()
