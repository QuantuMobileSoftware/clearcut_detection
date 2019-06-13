import argparse

parser = argparse.ArgumentParser()
arg = parser.add_argument

# arg('--loss_function', default='bce_dice')
# arg('--learning_rate', type=float, default=0.0001)

arg('--batch_size', type=int, default=8)
arg('--num_workers', type=int, default=4)
arg('--epochs', type=int, default=100)

arg('--logdir', default='../logs')
arg('--train_df', default='../data/train_df.csv')
arg('--val_df', default='../data/val_df.csv')
arg('--test_df', default='../data/test_df.csv')
arg('--dataset_path', default='../data')

arg('--image_size', type=int, default=224)

arg('--network', default='unet50')

arg(
    '--data_path', '-dp', dest='data_path',
    default='../data', help='Path to the data'
)
arg(
    '--save_path', '-sp', dest='save_path',
    default='../data',
    help='Path to directory where data will be stored'
)
arg(
    '--images_folder', '-imf', dest='images_folder',
    default='images',
    help='Name of folder where images are storing'
)
arg(
    '--masks_folder', '-mf', dest='masks_folder',
    default='masks',
    help='Name of folder where masks are storing'
)
arg(
    '--instances_folder', '-inf', dest='instances_folder',
    default='instance_masks',
    help='Name of folder where instances are storing'
)
arg(
    '--image_type', '-imt', dest='image_type',
    default='tiff',
    help='Type of image file'
)
arg(
    '--mask_type', '-mt', dest='mask_type',
    default='png',
    help='Type of mask file'
)
arg(
    '--instance_type', '-int', dest='instance_type',
    default='geojson',
    help='Type of instance file'
)
arg(
    '--channels', '-ch', dest='channels',
    default=['rgb', 'ndvi', 'ndvi_color', 'b2'],
    help='Channel list', type=list
)

args = parser.parse_args()
