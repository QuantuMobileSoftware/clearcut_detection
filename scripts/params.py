import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument

# arg('--loss_function', default='bce_dice')
# arg('--learning_rate', type=float, default=0.0001)

arg('--batch_size', type=int, default=8)
arg('--num_workers', type=int, default=4)
arg('--epochs', '-e', type=int, default=100)

arg('--logdir', default='../logs')
arg('--train_df', '-td', default='../../preprocessed_2016/filtered_4_seasons_train.csv')
arg('--val_df', '-vd', default='../../preprocessed_2016/filtered_4_seasons_test.csv')
arg('--dataset_path', '-dp', default='../../preprocessed_2016/')

arg('--img_size', '-is', type=int, default=224)

arg('--network', '-n', default='fpn50')

args = parser.parse_args()
