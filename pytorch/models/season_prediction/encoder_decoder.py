import torch
import torch.nn as nn

from segmentation_models_pytorch.base.model import Model


class EncoderDecoder(Model):

    def __init__(self, encoder, decoder, activation):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.season_model = nn.Sequential(
            nn.Linear(2048*7*7, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

        if callable(activation):
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError('Activation should be "sigmoid" or "softmax"')

    def forward(self, x):
        x = self.encoder(x)

        bottleneck = x[0]

        season_input = bottleneck.view(bottleneck.size()[0], -1)

        season_output = self.season_model(season_input)

        mask = self.decoder(x)

        return season_output, mask

    def predict(self, x):
        if self.training:
            self.eval()

        with torch.no_grad():
            season_input, mask = self.forward(x)
            mask = self.activation(mask)

        return season_input, mask
