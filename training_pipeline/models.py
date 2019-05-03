from keras.applications.vgg16 import VGG16
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers import Dense, Multiply, Add, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
# from inception_resnet_v2 import InceptionResNetV2
# from mobile_net_fixed import MobileNet
from resnet50_fixed import ResNet50, ResNet50_multi
from params import args
from sel_models.unets import (create_pyramid_features, conv_relu, prediction_fpn_block, conv_bn_relu, decoder_block_no_bn)
import numpy as np

def conv_block_simple(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = BatchNormalization(name=prefix + "_bn")(conv)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv


def conv_block_simple_no_bn(prevlayer, filters, prefix, strides=(1, 1)):
    conv = Conv2D(filters, (3, 3), padding="same", kernel_initializer="he_normal", strides=strides, name=prefix + "_conv")(prevlayer)
    conv = Activation('relu', name=prefix + "_activation")(conv)
    return conv


def cse_block(prevlayer, prefix):
    mean = Lambda(lambda xin: K.mean(xin, axis=[1, 2]))(prevlayer)
    lin1 = Dense(K.int_shape(prevlayer)[3]//2, name=prefix + 'cse_lin1', activation='relu')(mean)
    lin2 = Dense(K.int_shape(prevlayer)[3], name=prefix + 'cse_lin2', activation='sigmoid')(lin1)
    x = Multiply()([prevlayer, lin2])

    return x


def sse_block(prevlayer, prefix):
    # conv = Conv2D(K.int_shape(prevlayer)[3], (1, 1), padding="same", kernel_initializer="he_normal", activation='sigmoid', strides=(1, 1),
    #               name=prefix + "_conv")(prevlayer)
    conv = Conv2D(1, (1, 1), padding="same", kernel_initializer="he_normal", activation='sigmoid', strides=(1, 1),
                  name=prefix + "_conv")(prevlayer)
    conv = Multiply(name=prefix + "_mul")([prevlayer, conv])

    return conv


def csse_block(x, prefix):
    """
    Implementation of Concurrent Spatial and Channel ‘Squeeze & Excitation’ in Fully Convolutional Networks
    https://arxiv.org/abs/1803.02579
    """
    cse = cse_block(x, prefix)
    sse = sse_block(x, prefix)
    x = Add(name=prefix + "_csse_mul")([cse, sse])

    return x


"""
Unet with Mobile net encoder
Uses caffe preprocessing function
"""

def get_unet_resnet(input_shape):
    # model.add(Convolution2D(64, 3, 3, activation='relu'))
    # resh_conv = Conv2D()
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True

    conv1 = resnet_base.get_layer("activation_1").output
    conv2 = resnet_base.get_layer("activation_10").output
    conv3 = resnet_base.get_layer("activation_22").output
    conv4 = resnet_base.get_layer("activation_40").output
    conv5 = resnet_base.get_layer("activation_49").output

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    vgg = VGG16(input_shape=input_shape, input_tensor=resnet_base.input, include_top=False)
    for l in vgg.layers:
        l.trainable = False
    vgg_first_conv = vgg.get_layer("block1_conv2").output
    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input, vgg_first_conv], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    return model


def get_csse_unet_resnet(input_shape):
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True
    conv1 = resnet_base.get_layer("activation_1").output
    conv1 = csse_block(conv1, "csse_1")
    resnet_base.get_layer("max_pooling2d_1")(conv1)
    conv2 = resnet_base.get_layer("activation_10").output
    conv2 = csse_block(conv2, "csse_10")
    resnet_base.get_layer("res3a_branch2a")(conv2)
    conv3 = resnet_base.get_layer("activation_22").output
    conv3 = csse_block(conv3, "csse_22")
    resnet_base.get_layer("res4a_branch2a")(conv3)
    conv4 = resnet_base.get_layer("activation_40").output
    conv4 = csse_block(conv4, "csse_40")
    resnet_base.get_layer("res5a_branch2a")(conv4)
    conv5 = resnet_base.get_layer("activation_49").output
    conv5 = csse_block(conv5, "csse_49")
    resnet_base.get_layer("avg_pool")(conv5)

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")
    conv6 = csse_block(conv6, "csse_6")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")
    conv7 = csse_block(conv7, "csse_7")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")
    conv8 = csse_block(conv8, "csse_8")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")
    conv9 = csse_block(conv9, "csse_9")

    vgg = VGG16(input_shape=input_shape, input_tensor=resnet_base.input, include_top=False)
    for l in vgg.layers:
        l.trainable = False
    vgg_first_conv = vgg.get_layer("block1_conv2").output
    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input, vgg_first_conv], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = csse_block(conv10, "csse_o10")
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(resnet_base.input, x)
    return model


def csse_resnet50_fpn(input_shape, channels=1, activation="sigmoid"):
    # img_input = Input(input_shape)
    # conv_reshape = Conv2D(filters=3, kernel_size=(1, 1),
    #            strides=(1, 1), padding='same',
    #            kernel_initializer="he_normal",
    #            use_bias=False,
    #            name="input_conv")(img_input)
    # resnet_input = tuple([input_shape[0], input_shape[1], 3])
    # resnet_base = ResNet50(img_input, include_top=True)
    # resnet_base.load_weights(download_resnet_imagenet("resnet50"))
    #resnet_base = ResNet50(input_shape=input_shape, include_top=False)
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)
    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True
    # resnet_base = resnet_base(conv_reshape)
    conv1 = resnet_base.get_layer("activation_1").output
    conv1 = csse_block(conv1, "csse_1")
    resnet_base.get_layer("max_pooling2d_1")(conv1)
    conv2 = resnet_base.get_layer("activation_10").output
    conv2 = csse_block(conv2, "csse_10")
    resnet_base.get_layer("res3a_branch2a")(conv2)
    conv3 = resnet_base.get_layer("activation_22").output
    conv3 = csse_block(conv3, "csse_22")
    resnet_base.get_layer("res4a_branch2a")(conv3)
    conv4 = resnet_base.get_layer("activation_40").output
    conv4 = csse_block(conv4, "csse_40")
    resnet_base.get_layer("res5a_branch2a")(conv4)
    conv5 = resnet_base.get_layer("activation_49").output
    conv5 = csse_block(conv5, "csse_49")
    resnet_base.get_layer("avg_pool")(conv5)
    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            csse_block(prediction_fpn_block(P5, "P5", (8, 8)), "csse_P5"),
            csse_block(prediction_fpn_block(P4, "P4", (4, 4)), "csse_P4"),
            csse_block(prediction_fpn_block(P3, "P3", (2, 2)), "csse_P3"),
            csse_block(prediction_fpn_block(P2, "P2"), "csse_P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    x = Conv2D(channels, (1, 1), activation=activation, name="mask")(x)
    model = Model(resnet_base.input, x)
    return model


def csse_resnet50_fpn_multi(input_shape, channels=1, activation="sigmoid"):
    # img_input = Input(input_shape)
    # conv_reshape = Conv2D(filters=3, kernel_size=(1, 1),
    #            strides=(1, 1), padding='same',
    #            kernel_initializer="he_normal",
    #            use_bias=False,
    #            name="input_conv")(img_input)
    resnet_input = tuple([input_shape[0], input_shape[1], 3])
    # resnet_base = ResNet50(img_input, include_top=True)
    # resnet_base.load_weights(download_resnet_imagenet("resnet50"))
    #resnet_base = ResNet50(input_shape=input_shape, include_top=False)
    resnet_base = ResNet50(input_shape=input_shape, include_top=False, weights=None)
    resnet_base_we = ResNet50_multi(input_shape=resnet_input, include_top=False)

    conv_weights, conv_bias = resnet_base_we.layers[1].get_weights()
    # getting new_weights
    new_weights = np.random.normal(size=(7, 7, 4, 64), loc=0, scale=0.2)
    new_weights[:, :, :3, :] = conv_weights

    for i in resnet_base_we.layers:
        if i.name == 'conv1':
            resnet_base.get_layer(i.name).set_weights([new_weights, conv_bias])
        try:
            resnet_base.get_layer(i.name).set_weights(resnet_base_we.get_layer(i.name).get_weights())
        except:
            continue
    del resnet_base_we
    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True
    #resnet_base.get_layer('conv1')(conv_reshape)

    conv1 = resnet_base.get_layer("activation_1").output
    conv1 = csse_block(conv1, "csse_1")
    resnet_base.get_layer("max_pooling2d_1")(conv1)
    conv2 = resnet_base.get_layer("activation_10").output
    conv2 = csse_block(conv2, "csse_10")
    resnet_base.get_layer("res3a_branch2a")(conv2)
    conv3 = resnet_base.get_layer("activation_22").output
    conv3 = csse_block(conv3, "csse_22")
    resnet_base.get_layer("res4a_branch2a")(conv3)
    conv4 = resnet_base.get_layer("activation_40").output
    conv4 = csse_block(conv4, "csse_40")
    resnet_base.get_layer("res5a_branch2a")(conv4)
    conv5 = resnet_base.get_layer("activation_49").output
    conv5 = csse_block(conv5, "csse_49")
    resnet_base.get_layer("avg_pool")(conv5)
    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            csse_block(prediction_fpn_block(P5, "P5", (8, 8)), "csse_P5"),
            csse_block(prediction_fpn_block(P4, "P4", (4, 4)), "csse_P4"),
            csse_block(prediction_fpn_block(P3, "P3", (2, 2)), "csse_P3"),
            csse_block(prediction_fpn_block(P2, "P2"), "csse_P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    x = Conv2D(channels, (1, 1), activation=activation, name="mask")(x)

    model = Model(resnet_base.input, x)

    return model


def resnet50_fpn(input_shape, channels=1, activation="sigmoid"):
    # img_input = Input(input_shape)
    # resnet_base = ResNet50(img_input, include_top=True)
    # resnet_base.load_weights(download_resnet_imagenet("resnet50"))
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True
    # conv1 = resnet_base.get_layer("conv1_relu").output
    # conv2 = resnet_base.get_layer("res2c_relu").output
    # conv3 = resnet_base.get_layer("res3d_relu").output
    # conv4 = resnet_base.get_layer("res4f_relu").output
    # conv5 = resnet_base.get_layer("res5c_relu").output
    conv1 = resnet_base.get_layer("activation_1").output
    conv2 = resnet_base.get_layer("activation_10").output
    conv3 = resnet_base.get_layer("activation_22").output
    conv4 = resnet_base.get_layer("activation_40").output
    conv5 = resnet_base.get_layer("activation_49").output
    P1, P2, P3, P4, P5 = create_pyramid_features(conv1, conv2, conv3, conv4, conv5)
    x = concatenate(
        [
            prediction_fpn_block(P5, "P5", (8, 8)),
            prediction_fpn_block(P4, "P4", (4, 4)),
            prediction_fpn_block(P3, "P3", (2, 2)),
            prediction_fpn_block(P2, "P2"),
        ]
    )
    x = conv_bn_relu(x, 256, 3, (1, 1), name="aggregation")
    x = decoder_block_no_bn(x, 128, conv1, 'up4')
    x = UpSampling2D()(x)
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv1")
    x = conv_relu(x, 64, 3, (1, 1), name="up5_conv2")
    x = Conv2D(channels, (1, 1), activation=activation, name="mask")(x)
    model = Model(resnet_base.input, x)
    return model

def get_csse_hypercolumn_resnet(input_shape):
    resnet_base = ResNet50(input_shape=input_shape, include_top=False)

    if args.show_summary:
        resnet_base.summary()

    for l in resnet_base.layers:
        l.trainable = True
    conv1 = resnet_base.get_layer("activation_1").output
    conv1 = csse_block(conv1, "csse_1")
    resnet_base.get_layer("max_pooling2d_1")(conv1)
    conv2 = resnet_base.get_layer("activation_10").output
    conv2 = csse_block(conv2, "csse_10")
    resnet_base.get_layer("res3a_branch2a")(conv2)
    conv3 = resnet_base.get_layer("activation_22").output
    conv3 = csse_block(conv3, "csse_22")
    resnet_base.get_layer("res4a_branch2a")(conv3)
    conv4 = resnet_base.get_layer("activation_40").output
    conv4 = csse_block(conv4, "csse_40")
    resnet_base.get_layer("res5a_branch2a")(conv4)
    conv5 = resnet_base.get_layer("activation_49").output
    conv5 = csse_block(conv5, "csse_49")
    resnet_base.get_layer("avg_pool")(conv5)

    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")
    conv6 = csse_block(conv6, "csse_6")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 192, "conv7_1")
    conv7 = conv_block_simple(conv7, 192, "conv7_2")
    conv7 = csse_block(conv7, "csse_7")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")
    conv8 = csse_block(conv8, "csse_8")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")
    conv9 = csse_block(conv9, "csse_9")

    vgg = VGG16(input_shape=input_shape, input_tensor=resnet_base.input, include_top=False)
    for l in vgg.layers:
        l.trainable = False
    vgg_first_conv = vgg.get_layer("block1_conv2").output
    up10 = concatenate([UpSampling2D()(conv9), resnet_base.input, vgg_first_conv], axis=-1)
    conv10 = conv_block_simple(up10, 32, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = csse_block(conv10, "csse_o10")
    hyper = concatenate([conv10,
                         UpSampling2D(size=2)(conv9),
                         UpSampling2D(size=4)(conv8),
                         UpSampling2D(size=8)(conv7),
                         UpSampling2D(size=16)(conv6)], axis=-1)
    hyper = SpatialDropout2D(0.2)(hyper)
    # x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(hyper)
    x = Conv2D(1, (1, 1), name="no_activation_prediction", activation=None)(hyper)
    x = Activation('sigmoid', name="activation_prediction")(x)
    model = Model(resnet_base.input, x)
    return model


def get_simple_unet(input_shape):
    img_input = Input(input_shape)
    conv1 = conv_block_simple(img_input, 32, "conv1_1")
    conv1 = conv_block_simple(conv1, 32, "conv1_2")
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool1")(conv1)

    conv2 = conv_block_simple(pool1, 64, "conv2_1")
    conv2 = conv_block_simple(conv2, 64, "conv2_2")
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool2")(conv2)

    conv3 = conv_block_simple(pool2, 128, "conv3_1")
    conv3 = conv_block_simple(conv3, 128, "conv3_2")
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool3")(conv3)

    conv4 = conv_block_simple(pool3, 256, "conv4_1")
    conv4 = conv_block_simple(conv4, 256, "conv4_2")
    conv4 = conv_block_simple(conv4, 256, "conv4_3")

    up5 = concatenate([UpSampling2D()(conv4), conv3], axis=-1)
    conv5 = conv_block_simple(up5, 128, "conv5_1")
    conv5 = conv_block_simple(conv5, 128, "conv5_2")

    up6 = concatenate([UpSampling2D()(conv5), conv2], axis=-1)
    conv6 = conv_block_simple(up6, 64, "conv6_1")
    conv6 = conv_block_simple(conv6, 64, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv1], axis=-1)
    conv7 = conv_block_simple(up7, 32, "conv7_1")
    conv7 = conv_block_simple(conv7, 32, "conv7_2")

    conv7 = SpatialDropout2D(0.2)(conv7)

    prediction = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv7)
    model = Model(img_input, prediction)
    return model


def get_csse_unet(input_shape):
    img_input = Input(input_shape)
    conv1 = conv_block_simple(img_input, 32, "conv1_1")
    conv1 = conv_block_simple(conv1, 32, "conv1_2")
    conv1 = csse_block(conv1, "csse_1")
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool1")(conv1)

    conv2 = conv_block_simple(pool1, 64, "conv2_1")
    conv2 = conv_block_simple(conv2, 64, "conv2_2")
    conv2 = csse_block(conv2, "csse_2")
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool2")(conv2)

    conv3 = conv_block_simple(pool2, 128, "conv3_1")
    conv3 = conv_block_simple(conv3, 128, "conv3_2")
    conv3 = csse_block(conv3, "csse_3")
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), padding="same", name="pool3")(conv3)

    conv4 = conv_block_simple(pool3, 256, "conv4_1")
    conv4 = conv_block_simple(conv4, 256, "conv4_2")
    conv4 = conv_block_simple(conv4, 256, "conv4_3")

    up5 = concatenate([UpSampling2D()(conv4), conv3], axis=-1)
    conv5 = conv_block_simple(up5, 128, "conv5_1")
    conv5 = conv_block_simple(conv5, 128, "conv5_2")
    conv5 = csse_block(conv5, "csse_5")

    up6 = concatenate([UpSampling2D()(conv5), conv2], axis=-1)
    conv6 = conv_block_simple(up6, 64, "conv6_1")
    conv6 = conv_block_simple(conv6, 64, "conv6_2")
    conv6 = csse_block(conv6, "csse_6")

    up7 = concatenate([UpSampling2D()(conv6), conv1], axis=-1)
    conv7 = conv_block_simple(up7, 32, "conv7_1")
    conv7 = conv_block_simple(conv7, 32, "conv7_2")
    conv7 = csse_block(conv7, "csse_7")

    conv7 = SpatialDropout2D(0.2)(conv7)

    prediction = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv7)
    model = Model(img_input, prediction)
    return model
"""
Unet with Mobile net encoder
Uses the same preprocessing as in Inception, Xception etc. (imagenet_utils.preprocess_input with mode 'tf' in new Keras version)
"""


def get_unet_mobilenet(input_shape):
    base_model = MobileNet(include_top=False, input_shape=input_shape)

    conv1 = base_model.get_layer('conv_pw_1_relu').output
    conv2 = base_model.get_layer('conv_pw_3_relu').output
    conv3 = base_model.get_layer('conv_pw_5_relu').output
    conv4 = base_model.get_layer('conv_pw_11_relu').output
    conv5 = base_model.get_layer('conv_pw_13_relu').output
    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 256, "conv7_1")
    conv7 = conv_block_simple(conv7, 256, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 192, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 96, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = concatenate([UpSampling2D()(conv9), base_model.input], axis=-1)
    conv10 = conv_block_simple(up10, 48, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.2)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(base_model.input, x)
    return model


"""
Unet with Inception Resnet V2 encoder
Uses the same preprocessing as in Inception, Xception etc. (imagenet_utils.preprocess_input with mode 'tf' in new Keras version)
"""


def get_unet_inception_resnet_v2(input_shape):
    base_model = InceptionResNetV2(include_top=False, input_shape=input_shape)
    conv1 = base_model.get_layer('activation_3').output
    conv2 = base_model.get_layer('activation_5').output
    conv3 = base_model.get_layer('block35_10_ac').output
    conv4 = base_model.get_layer('block17_20_ac').output
    conv5 = base_model.get_layer('conv_7b_ac').output
    up6 = concatenate([UpSampling2D()(conv5), conv4], axis=-1)
    conv6 = conv_block_simple(up6, 256, "conv6_1")
    conv6 = conv_block_simple(conv6, 256, "conv6_2")

    up7 = concatenate([UpSampling2D()(conv6), conv3], axis=-1)
    conv7 = conv_block_simple(up7, 256, "conv7_1")
    conv7 = conv_block_simple(conv7, 256, "conv7_2")

    up8 = concatenate([UpSampling2D()(conv7), conv2], axis=-1)
    conv8 = conv_block_simple(up8, 128, "conv8_1")
    conv8 = conv_block_simple(conv8, 128, "conv8_2")

    up9 = concatenate([UpSampling2D()(conv8), conv1], axis=-1)
    conv9 = conv_block_simple(up9, 64, "conv9_1")
    conv9 = conv_block_simple(conv9, 64, "conv9_2")

    up10 = concatenate([UpSampling2D()(conv9), base_model.input], axis=-1)
    conv10 = conv_block_simple(up10, 48, "conv10_1")
    conv10 = conv_block_simple(conv10, 32, "conv10_2")
    conv10 = SpatialDropout2D(0.4)(conv10)
    x = Conv2D(1, (1, 1), activation="sigmoid", name="prediction")(conv10)
    model = Model(base_model.input, x)
    return model


def get_vgg_7conv(input_shape):
    img_input = Input(input_shape)
    vgg16_base = VGG16(input_tensor=img_input, include_top=False)
    for l in vgg16_base.layers:
        l.trainable = True
    conv1 = vgg16_base.get_layer("block1_conv2").output
    conv2 = vgg16_base.get_layer("block2_conv2").output
    conv3 = vgg16_base.get_layer("block3_conv3").output
    pool3 = vgg16_base.get_layer("block3_pool").output

    conv4 = Conv2D(384, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block4_conv1")(pool3)
    conv4 = Conv2D(384, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block4_conv2")(conv4)
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(conv4)

    conv5 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block5_conv1")(pool4)
    conv5 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block5_conv2")(conv5)
    pool5 = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(conv5)

    conv6 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block6_conv1")(pool5)
    conv6 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block6_conv2")(conv6)
    pool6 = MaxPooling2D((2, 2), strides=(2, 2), name='block6_pool')(conv6)

    conv7 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block7_conv1")(pool6)
    conv7 = Conv2D(512, (3, 3), activation="relu", padding='same', kernel_initializer="he_normal", name="block7_conv2")(conv7)

    up8 = concatenate([Conv2DTranspose(384, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv7), conv6], axis=3)
    conv8 = Conv2D(384, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up8)

    up9 = concatenate([Conv2DTranspose(256, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv8), conv5], axis=3)
    conv9 = Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up9)

    up10 = concatenate([Conv2DTranspose(192, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv9), conv4], axis=3)
    conv10 = Conv2D(192, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up10)

    up11 = concatenate([Conv2DTranspose(128, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv10), conv3], axis=3)
    conv11 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up11)

    up12 = concatenate([Conv2DTranspose(64, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv11), conv2], axis=3)
    conv12 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up12)

    up13 = concatenate([Conv2DTranspose(32, (3, 3), activation="relu", kernel_initializer="he_normal", strides=(2, 2), padding='same')(conv12), conv1], axis=3)
    conv13 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding='same')(up13)

    conv13 = Conv2D(1, (1, 1))(conv13)
    conv13 = Activation("sigmoid")(conv13)
    model = Model(img_input, conv13)
    return model


def make_model(input_shape):
    network = args.network
    if network == 'resnet50':
        return get_unet_resnet(input_shape)
    elif network == 'csse_resnet50':
        return get_csse_unet_resnet(input_shape)
    elif network == 'hypercolumn_resnet':
        return get_csse_hypercolumn_resnet(input_shape)
    elif network == 'inception_resnet_v2':
        return get_unet_inception_resnet_v2(input_shape)
    elif network == 'mobilenet':
        return get_unet_mobilenet(input_shape)
    elif network == 'vgg':
        return get_vgg_7conv(input_shape)
    elif network == 'simple_unet':
        return get_simple_unet(input_shape)
    elif network == 'csse_unet':
        return get_csse_unet(input_shape)
    elif network == 'resnet50_fpn':
        return resnet50_fpn(input_shape, channels=1, activation="sigmoid")
    elif network == 'csse_resnet50_fpn':
        return csse_resnet50_fpn(input_shape, channels=1, activation="sigmoid")
    elif network == 'csse_resnet50_fpn_multi':
        return csse_resnet50_fpn_multi(input_shape, channels=1, activation="sigmoid")
    else:
        raise ValueError("Unknown network")
