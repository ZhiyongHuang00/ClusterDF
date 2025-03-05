# -*- coding: utf-8 -*-
'''
@Time    : 2024/9/6 10:55
@Author  : HuangZhiyong
@email   : 1524338616@qq.com
@File    : Model_60s.py
@Function: {}:
'''
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Reshape, Conv2DTranspose
from tensorflow.keras import backend as K


def set_global_determinism(seed):
    import os
    import random
    import numpy as np
    import tensorflow as tf

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    tf.compat.v1.keras.backend.set_session(sess)


def set_seeds(seed=87):
    import random
    import numpy as np
    import tensorflow as tf

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def CAE(input_shape=(64, 600, 1), filters=[8, 16, 32, 64, 128], encoding_filters=32, set_seed=[False, 87],
        summary=True):
    if set_seed[0] is True:
        set_global_determinism(seed=set_seed[1])
        set_seeds(seed=set_seed[1])

    inp = Input(shape=input_shape, name="original_img")
    x = Conv2D(filters=filters[0], kernel_size=3, strides=2, activation="relu", padding="same", name="Conv_1")(inp)
    x = Conv2D(filters=filters[1], kernel_size=3, strides=2, activation="relu", padding="same", name="Conv_2")(x)
    # x = Conv2D(filters=filters[2], kernel_size=3, strides=2, activation="relu", padding="same", name="Conv_3")(x)
    # x = Conv2D(filters=filters[3], kernel_size=3, strides=2, activation="relu", padding="same", name="Conv_4")(x)
    x = Conv2D(filters=encoding_filters, kernel_size=3, strides=2, activation="relu", padding="same", name="Conv_5")(x)

    shape_before_flattening = K.int_shape(x)
    x = Flatten(input_shape=shape_before_flattening[1:], name="Flatten")(x)
    encoder_output = Dense(encoding_filters, activation="linear", name="encoded")(x)

    encoder = Model(inp, encoder_output, name="encoder")

    # Decoder
    decoder_input = encoder_output
    x = Dense(shape_before_flattening[1] * shape_before_flattening[2] * shape_before_flattening[3], activation="relu")(decoder_input)
    x = Reshape((shape_before_flattening[1:]))(x)

    x = Conv2DTranspose(filters=encoding_filters, kernel_size=3, strides=2, activation="relu", padding="same")(x)
    # x = Conv2DTranspose(filters=filters[3], kernel_size=3, strides=2, activation="relu", padding="same")(x)
    # x = Conv2DTranspose(filters=filters[2], kernel_size=3, strides=2, activation="relu", padding="same")(x)
    x = Conv2DTranspose(filters=filters[1], kernel_size=3, strides=2, activation="relu", padding="same")(x)
    x = Conv2DTranspose(filters=filters[0], kernel_size=3, strides=2, activation="relu", padding="same")(x)
    decoder_output = Conv2DTranspose(filters=input_shape[2], kernel_size=3, strides=1, padding="same", name="decoded", activation="linear")(x)

    autoencoder = Model(inputs=inp, outputs=decoder_output, name="autoencoder")
    decoder = Model(inputs=decoder_input,
                    outputs=decoder_output,
                    name="decoder")
    if summary:
        autoencoder.summary()

    return encoder, decoder, autoencoder

# Example usage
encoder, decoder, autoencoder = CAE()

