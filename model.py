#-*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
from utils.utils_tool import logger


text_scale = 512

from FPN import FPN


class resize_image(layers.Layer):

    def __init__(self, target_tensor_shape, target_int_shape, *args, **kwargs):
        self.target_tensor_shape = target_tensor_shape
        self.target_int_shape = target_int_shape
        super(resize_image, self).__init__(*args, **kwargs)

    def call(self, input_tensor, **kwargs):
        print(self.target_int_shape)
        return tf.image.resize(input_tensor, (self.target_tensor_shape[0], self.target_tensor_shape[1]),
                                      method=tf.image.ResizeMethod.BILINEAR)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],) + (self.target_int_shape[0], self.target_int_shape[1]) + (input_shape[-1],)


#TODO:bilinear or nearest_neighbor?
def unpool(inputs, rate):
    return tf.compat.v1.image.resize_bilinear(inputs, size=[tf.shape(inputs)[1]*rate,  tf.shape(inputs)[2]*rate])


def mean_image_subtraction(images, means=[123.68, 116.78, 103.94]):
    '''
    image normalization
    :param images:
    :param means:
    :return:
    '''
    num_channels = images.get_shape().as_list()[-1]
    if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
    channels = tf.split(axis=3, num_or_size_splits=num_channels, value=images)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=3, values=channels)


def conv_bn_relu(input_tensor, filters, kernel_size=3, bn=True,
                relu=True, isTraining=True, weight_decay=1e-5):
    '''
    conv2d + bn + relu
    notice :
        isTraining : if finetune model should set False
        ? wether add l2 regularizer?
    '''
    x = layers.Conv2D(filters, kernel_size, strides=(1, 1),
               padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input_tensor)
    if(bn):
        x = layers.BatchNormalization(axis=-1)(x)

    if(relu):
        x = layers.Activation('relu')(x)
    return x


def FC_SN(PN):

    # we fuse the four feature maps to get feature map F with 1024 channels via the function
    #C(·) as: F = C(P2, P3, P4, P5) = P2 || Up×2(P3) || Up×4(P4) || Up×8 (P5), where “k” refers to the concatenation and Up×2
    # # #(·), Up×4 (·), Up×8 (·) refer to 2, 4, 8 times upsampling
    # P2 = PN[-1]
    # t_tensor_shape = tf.shape(P2)[1:3]
    # t_int_shape = tf.keras.backend.int_shape(P2)[1:3]
    #
    # for i in range(len(PN)-1):
    #     PN[i] = resize_image(t_tensor_shape, t_int_shape)(PN[i])

    # print(PN.shape)
    # unpool sample P
    P_concat = []
    for i in range(3, 0, -1):
        P_concat.append(unpool(PN[i], 2 ** i))
    P_concat.append(PN[0])
    F = layers.concatenate(P_concat, axis=-1)

    #F is fed into Conv(3, 3)-BN-ReLU layers and is reduced to 256 channels.
    F = conv_bn_relu(F, 256)

    #Next, it passes through multiple Conv(1, 1)-Up-Sigmoid layers and produces n segmentation results
    #S1, S2, ..., Sn.
    SN = layers.Conv2D(6, (1, 1))(F)
    #
    # scale = 1
    # # if(config.ns == 2):
    # #     scale = 1
    #
    # #new_shape = t_tensor_shape
    # #new_shape *= tf.constant(np.array([scale, scale], dtype='int32'))
    # if t_int_shape[0] is None:
    #     new_height = None
    # else:
    #     new_height = t_int_shape[0] * scale
    # if t_int_shape[1] is None:
    #     new_width = None
    # else:
    #     new_width = t_int_shape[1] * scale
    #
    # SN = resize_image(t_tensor_shape, (new_height, new_width))(SN)
    SN = layers.Activation('sigmoid')(SN)

    return SN


class PSEnet():
    def __init__(self):
        self.model = self.build()

    def build(self):
        input_image = layers.Input(shape=[None, None, 3], name='input_image')

        P2, P3, P4, P5, _ = FPN(input_image, 'resnet50', stage5=True, train_bn=False)
        PN = [P2, P3, P4, P5]
        seg_S_pred = FC_SN(PN)

        inputs = [input_image]
        outputs = [seg_S_pred]
        model = tf.keras.Model(inputs, outputs, name='psenet')
        return model


def dice_coefficient(y_true_cls, y_pred_cls,
                     training_mask):
    '''
    dice loss
    :param y_true_cls: ground truth
    :param y_pred_cls: predict
    :param training_mask:
    :return:
    '''
    eps = 1e-5
    intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
    union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
    dice = 2 * intersection / union
    loss = 1. - dice
    # tf.summary.scalar('classification_dice_loss', loss)
    return dice, loss


def loss(y_true_cls, y_pred_cls,
         training_mask):
    g1, g2, g3, g4, g5, g6 = tf.split(value=y_true_cls, num_or_size_splits=6, axis=3)
    s1, s2, s3, s4, s5, s6 = tf.split(value=y_pred_cls, num_or_size_splits=6, axis=3)
    Gn = [g1, g2, g3, g4, g5, g6]
    Sn = [s1, s2, s3, s4, s5, s6]
    _, Lc = dice_coefficient(Gn[5], Sn[5], training_mask=training_mask)
    tf.summary.scalar('Lc_loss', Lc)

    one = tf.ones_like(Sn[5])
    zero = tf.zeros_like(Sn[5])
    W = tf.where(Sn[5] >= 0.5, x=one, y=zero)
    D = 0
    for i in range(5):
        di, _ = dice_coefficient(Gn[i]*W, Sn[i]*W, training_mask=training_mask)
        D += di
    Ls = 1-D/5.
    tf.summary.scalar('Ls_loss', Ls)
    lambda_ = 0.7
    L = lambda_*Lc + (1-lambda_)*Ls
    return L




