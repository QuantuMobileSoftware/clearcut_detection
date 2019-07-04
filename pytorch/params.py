import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument

arg('--batch_size', type=int, default=8)
arg('--num_workers', type=int, default=4)
arg('--epochs', '-e', type=int, default=100)
arg('--folds', '-f', type=int, default=None)
arg('--pseudolabel_iter', '-pi', type=int, default=2)

arg('--logdir', default='../logs')
arg('--train_df', '-td', default='../data/train_df.csv')
arg('--val_df', '-vd', default='../data/val_df.csv')
arg('--test_df', '-ttd', default='../data/test_df.csv')
arg('--dataset_path', '-dp', default='../data/input', help='Path to the data')
arg('--unlabeled_data', '-ud', default='../data/unlabeled_data')
arg('--model_weights_path', '-mwp', default='../weights/resnet50-19c8e357.pth')

arg('--image_size', '-is', type=int, default=224)

arg('--network', '-n', default='unet50')

arg(
    '--channels', '-ch',
    default=[
        'rgb', 'ndvi', 'ndvi_color',
        'b2', 'b3', 'b4', 'b8'
    ], nargs='+', help='Channels list')

args = parser.parse_args()
