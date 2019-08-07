from segmentation_models_pytorch.base import EncoderDecoder
from segmentation_models_pytorch.encoders import get_encoder
from clearcut_research.pytorch.models.autoencoder.decoder import UnetDecoder


class Autoencoder_Unet(EncoderDecoder):

    def __init__(
            self,
            encoder_name='resnet50',
            encoder_weights='imagenet',
            decoder_use_batchnorm=True,
            decoder_channels=(256, 128, 64, 32, 16),
            classes=3,
            activation='sigmoid',
            center=False,  # usefull for VGG models
    ):
        encoder = get_encoder(
            encoder_name,
            encoder_weights='imagenet'
        )

        decoder = UnetDecoder(
            encoder_channels=encoder.out_shapes,
            decoder_channels=decoder_channels,
            final_channels=classes,
            use_batchnorm=decoder_use_batchnorm,
            center=center,
        )

        super().__init__(encoder, decoder, activation)

        self.name = 'u-{}'.format(encoder_name)
