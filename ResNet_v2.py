import tensorflow as tf
import numpy as np
import errno
import os
from tensorflow.python.keras.api._v2.keras import layers, Sequential, regularizers
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation,add, Dropout, MaxPooling2D, LeakyReLU
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten, Reshape, DepthwiseConv2D,GlobalAveragePooling2D, Permute
from tensorflow.keras.models import Model
from tensorflow.keras import models, backend

# network config
# stack_n = 3  # layers = stack_n * 6 + 2
weight_decay = 5e-4

def VGG16(input_shape, n_classes):
    model_A, x = None, None

    inputs = Input(input_shape)
    if len(input_shape) == 2:
        y = Reshape((input_shape[0], input_shape[1], 1))(inputs)
    else:
        y = Reshape(input_shape)(inputs)
    # model = models.Sequential()
    y = Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=input_shape,
               kernel_regularizer=regularizers.l2(weight_decay))(y)
    y = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(y)
    y = MaxPooling2D((2, 2))(y)

    y = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(y)
    y = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(y)
    y = MaxPooling2D((2, 2))(y)

    y = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(y)
    y = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(y)
    y = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(y)
    y = MaxPooling2D((2, 2))(y)

    y = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(y)
    y = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(y)
    y = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(y)
    y = MaxPooling2D((2, 2))(y)

    y = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(y)
    y = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(y)
    y = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(weight_decay))(y)

    y = Flatten()(y) # 2*2*512
    y = Dense(4096, activation='relu')(y)
    y = Dropout(0.5)(y)
    y = Dense(4096, activation='relu')(y)
    y = Dropout(0.5)(y)
    logits = Dense(n_classes, activation=None)(y)
    valid_logit = Dense(units=1, activation=None, use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l2(1e-3), name='dense-valid-1')(y)
    # 对于原始的完整模型而言，这些层不会被更新
    valid_logit.trainable = False
    y = Activation("softmax")(logits)
    valid = Activation("sigmoid")(valid_logit)
    # outputs = Activation('softmax')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=[y, valid])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def residual_block(inputs, channels, strides=(1, 1)):
    net = BatchNormalization(momentum=0.9, epsilon=1e-5)(inputs)
    net = Activation('relu')(net)

    if strides == (1, 1):
        shortcut = inputs
    else:
        shortcut = Conv2D(channels, (1, 1), strides=strides)(net)

    net = Conv2D(channels, (3, 3), padding='same', strides=strides)(net)
    net = BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
    net = Activation('relu')(net)
    net = Conv2D(channels, (3, 3), padding='same')(net)

    net = add([net, shortcut])
    return net

def ResNet(inputs_shape, stack_n, n_classes):
    inputs = Input(shape=inputs_shape)
    net = Conv2D(16, (3, 3), padding='same')(inputs)

    for i in range(stack_n):
        net = residual_block(net, 16)

    net = residual_block(net, 32, strides=(2, 2))
    for i in range(stack_n - 1):
        net = residual_block(net, 32)

    net = residual_block(net, 64, strides=(2, 2))
    for i in range(stack_n - 1):
        net = residual_block(net, 64)

    net = BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
    net = Activation('relu')(net)
    net = AveragePooling2D(8, 8)(net)
    net = Flatten()(net)
    logits = Dense(n_classes, activation=None)(net)
    valid_logit = Dense(units=1, activation=None, use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l2(1e-3), name='dense-valid-1')(net)
    # 对于原始的完整模型而言，这些层不会被更新
    valid_logit.trainable = False
    y = Activation("softmax")(logits)
    valid = Activation("sigmoid")(valid_logit)
    # outputs = Activation('softmax')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=[y, valid])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

"""
======================================================================
ResNet20,32,44,56,VGG16
======================================================================
"""

################################################################################################
#  定义MobileNet v2
"""
# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""

# import tensorflow as tf
# Define ReLU6 activation
relu6 = tf.keras.layers.ReLU(6.)

def _conv_block(inputs, filters, kernel, strides):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.
    # Returns
        Output tensor.
    """

    x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
    x = BatchNormalization()(x)
    return relu6(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.
    # Returns
        Output tensor.
    """

    tchannel = inputs.shape[-1] * t

    x = _conv_block(inputs, tchannel, (1, 1), (1, 1))

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = relu6(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    if r:
        x = add([x, inputs])
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.
    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x


def MobileNetV2(input_shape, n_classes, plot_model=False):
    """MobileNetv2
    This function defines a MobileNetv2 architecture.
    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        n_classes: Integer, number of classes.
        plot_model: Boolean, whether to plot model architecture or not
    # Returns
        MobileNetv2 model.
    """

    inputs = Input(shape=input_shape, name='input')
    x = _conv_block(inputs, 32, (3, 3), strides=(2, 2))

    x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
    x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
    x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)

    x = _conv_block(x, 1280, (1, 1), strides=(1, 1))
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, 1280))(x)
    x = Dropout(0.3, name='Dropout')(x)
    x = Conv2D(n_classes, (1, 1), padding='same')(x)
    x = Reshape((n_classes,), name='output')(x)
    logits = Dense(n_classes, activation=None)(x)
    valid_logit = Dense(units=1, activation=None, use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l2(1e-3), name='dense-valid-1')(x)
    # 对于原始的完整模型而言，这些层不会被更新
    valid_logit.trainable = False
    y = Activation('softmax', name='final_activation')(logits)
    valid = Activation('sigmoid')(valid_logit)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=[y, valid])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

################################################################################################
#  定义ShuffleNet v2
def channel_shuffle(inputs, num_groups):
    n, h, w, c = inputs.shape
    x_reshaped = tf.reshape(inputs, [-1, h, w, num_groups, c // num_groups])
    x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
    output = tf.reshape(x_transposed, [-1, h, w, c])

    return output


def conv(inputs, filters, kernel_size, strides=1):
    x = Conv2D(filters, kernel_size, strides, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    return x


def depthwise_conv_bn(inputs, kernel_size, strides=1):
    x = DepthwiseConv2D(kernel_size=kernel_size,
                                        strides=strides,
                                        padding='same')(inputs)
    x = BatchNormalization()(x)

    return x


def ShuffleNetUnitA(inputs, out_channels):
    shortcut, x = tf.split(inputs, 2, axis=-1)

    x = conv(inputs, out_channels // 2, kernel_size=1, strides=1)
    x = depthwise_conv_bn(x, kernel_size=3, strides=1)
    x = conv(x, out_channels // 2, kernel_size=1, strides=1)

    x = tf.concat([shortcut, x], axis=-1)
    x = channel_shuffle(x, 2)

    return x


def ShuffleNetUnitB(inputs, out_channels):
    shortcut = inputs

    in_channels = inputs.shape[-1]

    x = conv(inputs, out_channels // 2, kernel_size=1, strides=1)
    x = depthwise_conv_bn(x, kernel_size=3, strides=2)
    x = conv(x, out_channels - in_channels, kernel_size=1, strides=1)

    shortcut = depthwise_conv_bn(shortcut, kernel_size=3, strides=2)
    shortcut = conv(shortcut, in_channels, kernel_size=1, strides=1)

    output = tf.concat([shortcut, x], axis=-1)
    output = channel_shuffle(output, 2)

    return output


def stage(inputs, out_channels, n):
    x = ShuffleNetUnitB(inputs, out_channels)

    for _ in range(n):
        x = ShuffleNetUnitA(x, out_channels)

    return x


def ShuffleNet_v2(input_shape, first_stage_channels, num_groups, n_classes):
    inputs = Input(shape=input_shape, name='input')

    x = Conv2D(filters=24, kernel_size=3, strides=2, padding='same')(inputs)
    x = MaxPooling2D(pool_size=3, strides=2, padding='same')(x)

    x = stage(x, first_stage_channels, n=3)
    x = stage(x, first_stage_channels * 2, n=7)
    x = stage(x, first_stage_channels * 4, n=3)

    x = Conv2D(filters=1024, kernel_size=1, strides=1, padding='same')(x)
    x = GlobalAveragePooling2D()(x)
    logits = Dense(n_classes, activation=None)(x)
    valid_logit = Dense(units=1, activation=None, use_bias=False,
                        kernel_regularizer=tf.keras.regularizers.l2(1e-3), name='dense-valid-1')(x)
    # 对于原始的完整模型而言，这些层不会被更新
    valid_logit.trainable = False
    y = Activation('softmax', name='final_activation')(logits)
    valid = Activation('sigmoid')(valid_logit)
    # y = Reshape((n_classes,), name='output')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=[y, valid])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model

def build_model(input_shape, n_classes, n_parties):
    parties = []
    n1 = 3  # ResNet20
    # n2 = 5  # ResNet32
    # n3 = 7  # ResNet44
    n4 = 9  # ResNet56
    for i in range(n_parties):
        if i < 3:  # 0,1,2
            tmp = ResNet(input_shape, n1, n_classes)
            print("n={0}, model {1} : ".format(n1, i))
            print(tmp.summary())
            parties.append(tmp)

            del tmp
        elif 3 <= i < 6:  # 3,4,5
            tmp = ResNet(input_shape, n4, n_classes)
            print("n={0}, model {1} : ".format(n4, i))
            print(tmp.summary())
            parties.append(tmp)

            del tmp

        elif 6 <= i < 8:  # 6,7
            # del tmp
            tmp = MobileNetV2(input_shape, n_classes)
            print("model {0} : ".format(i))
            print(tmp.summary())
            parties.append(tmp)

            del tmp
        else:
            # elif i == 8,9:
            tmp = ShuffleNet_v2(input_shape, 144, 1, n_classes)
            print("model {0} : ".format(i))
            # tmp = ResNet(input_shape, n2, n_classes)
            # print("n={0}, model {1} : ".format(n2, i))
            print(tmp.summary())
            parties.append(tmp)

            del tmp
    return parties

# def build_model(input_shape, n_classes, n_parties):
#     parties = []
#     n1 = 3  # ResNet20
#     # n2 = 5  # ResNet32
#     # n3 = 7  # ResNet44
#     # n4 = 9  # ResNet56
#     # n5 = 18  # ResNet110
#     for i in range(n_parties):
#         if i < 4:  # 0,1,2,3
#             tmp = ResNet(input_shape, n1, n_classes)
#             print("n={0}, model {1} : ".format(n1, i))
#             print(tmp.summary())
#             parties.append(tmp)
#
#             del tmp
#         # elif 4 <= i < 7:  # 4,5,6
#         else:
#             tmp = MobileNetV2(input_shape, n_classes)
#             print("model {0} : ".format(i))
#             print(tmp.summary())
#             parties.append(tmp)
#
#             del tmp
#         # else:
#             # elif i == 7,8,9:
#             # tmp = ShuffleNet_v2(input_shape, 144, 1, n_classes)
#             # print("model {0} : ".format(i))
#             # tmp = ResNet(input_shape, n2, n_classes)
#             # print("n={0}, model {1} : ".format(n2, i))
#             # print(tmp.summary())
#             # parties.append(tmp)
#
#             # del tmp
#     return parties

def remove_last_layer(model, loss):
    """
    Input: Keras model, a classification model whose last layer is a softmax activation
    Output: Keras model, the same model with the last softmax activation layer removed,
        while keeping the same parameters
    """
    # print("-1:", model.layers[-1].output) # sigmoid 输出
    # print("-2:", model.layers[-2].output) # softmax 输出
    # print("-3:", model.layers[-3].output) # Dense(1) 输出
    # print("-4:", model.layers[-4].output) # Dense(16) 输出

    new_model = Model(inputs=model.inputs, outputs=[model.layers[-4].output, model.layers[-1].output])
    # new_model.set_weights(model.get_weights())
    """
        由于删除了softmax层，因此new_model的weights下标产生变化，这里手动实现set_weights的操作
    """
    weights = model.get_weights()
    new_model = modify_set_weights(new_model, weights)
    new_model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                      loss=loss, loss_weights=[0.8, 0.2])

    return new_model


def train_models(models, X_train, y_train, X_test, y_test,
                 save_dir="./", save_names=None,
                 early_stopping=True, min_delta=0.001, patience=3,
                 batch_size=128, epochs=20, is_shuffle=True, verbose=1
                 ):
    '''
    Train an array of models on the same dataset.
    We use early termination to speed up training.
    '''

    resulting_val_acc = []
    record_result = []
    for n, model in enumerate(models):
        print("Training model ", n)
        # if early_stopping:
        #     model.fit(X_train, y_train,
        #               validation_data = [X_test, y_test],
        #               callbacks=[EarlyStopping(monitor='val_accuracy', min_delta=min_delta, patience=patience)],
        #               batch_size = batch_size, epochs = epochs, shuffle=is_shuffle, verbose = verbose
        #              )
        # else:
        #     model.fit(X_train, y_train,
        #               validation_data = [X_test, y_test],
        #               batch_size = batch_size, epochs = epochs, shuffle=is_shuffle, verbose = verbose
        #              )
        #
        # resulting_val_acc.append(model.history.history["val_accuracy"][-1])
        # record_result.append({"train_acc": model.history.history["accuracy"],
        #                       "val_acc": model.history.history["val_accuracy"],
        #                       "train_loss": model.history.history["loss"],
        #                       "val_loss": model.history.history["val_loss"]})

        if save_dir is not None:
            save_dir_path = os.path.abspath(save_dir)
            # make dir
            try:
                os.makedirs(save_dir_path)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

            if save_names is None:
                file_name = save_dir + "model_{0}".format(n) + ".h5"
            else:
                file_name = save_dir + save_names[n] + ".h5"
            model.save(file_name)

    print("pre-train accuracy: ")
    print(resulting_val_acc)

    return record_result


def modify_set_weights(new_model, weights):
    if len(new_model.weights) != len(weights):
        raise ValueError('You called `set_weights(weights)` on layer "' +
                         new_model.name + '" with a  weight list of length ' +
                         str(len(weights)) + ', but the layer was expecting ' +
                         str(len(new_model.weights)) + ' weights. Provided weights: ' +
                         str(weights)[:50] + '...')
    if not new_model.weights:
        return
    weight_value_tuples = []
    param_values = backend.batch_get_value(new_model.weights)
    r = 0
    l = len(new_model.weights)
    if l == 331:  # for ShuffleNet_v2
        for pv, p, w in zip(param_values, new_model.weights, weights):
            # if r == l - 3:  # 手动调整最后三层的weight的赋值
            #     new_3 = w
            #     old_3 = p
            #     # layer_18 = pv
            #     # weight_value_tuples.append((w, p))
            # elif r == l - 2:
            #     new_2 = w
            #     old_2 = p
            # elif r == l - 1:
            #     new_1 = w
            #     old_1 = p
            #     # layer_19 = pv
            #     weight_value_tuples.append((old_3, new_1))
            #     weight_value_tuples.append((old_2, new_3))
            #     weight_value_tuples.append((old_1, new_2))
            #     break
            if pv.shape != w.shape:
                raise ValueError('Layer weight shape ' + str(pv.shape) +
                                 ' not compatible with '
                                 'provided weight shape ' + str(w.shape))
            else:
                weight_value_tuples.append((p, w))
            r += 1
        backend.batch_set_value(weight_value_tuples)
    else:
        for pv, p, w in zip(param_values, new_model.weights, weights):
            if r == l - 3:  # 手动调整最后三层的weight的赋值
                new_3 = w
                old_3 = p
                # layer_18 = pv
                # weight_value_tuples.append((w, p))
            elif r == l - 2:
                new_2 = w
                old_2 = p
            elif r == l - 1:
                new_1 = w
                old_1 = p
                # layer_19 = pv
                weight_value_tuples.append((old_3, new_1))
                weight_value_tuples.append((old_2, new_3))
                weight_value_tuples.append((old_1, new_2))
                break
            elif pv.shape != w.shape:
                raise ValueError('Layer weight shape ' + str(pv.shape) +
                                 ' not compatible with '
                                 'provided weight shape ' + str(w.shape))
            else:
                weight_value_tuples.append((p, w))
            r += 1
        backend.batch_set_value(weight_value_tuples)
    return new_model


def modify_set_weights_fro_classifier(new_model, weights):
    if len(new_model.weights) != len(weights):
        raise ValueError('You called `set_weights(weights)` on layer "' +
                         new_model.name + '" with a  weight list of length ' +
                         str(len(weights)) + ', but the layer was expecting ' +
                         str(len(new_model.weights)) + ' weights. Provided weights: ' +
                         str(weights)[:50] + '...')
    if not new_model.weights:
        return
    weight_value_tuples = []
    param_values = backend.batch_get_value(new_model.weights)
    r = 0
    l = len(new_model.weights)
    for pv, p, w in zip(param_values, new_model.weights, weights):
        if r == l - 3:  # 手动调整最后三层的weight的赋值
            new_3 = w
            old_3 = p
            # layer_18 = pv
            # weight_value_tuples.append((w, p))
        elif r == l - 2:
            new_2 = w
            old_2 = p
        elif r == l - 1:
            new_1 = w
            old_1 = p
            # layer_19 = pv
            weight_value_tuples.append((old_3, new_2))
            weight_value_tuples.append((old_2, new_1))
            weight_value_tuples.append((old_1, new_3))
            break
        elif pv.shape != w.shape:
            raise ValueError('Layer weight shape ' + str(pv.shape) +
                             ' not compatible with '
                             'provided weight shape ' + str(w.shape))
        else:
            weight_value_tuples.append((p, w))
        r += 1
    backend.batch_set_value(weight_value_tuples)
    return new_model

def build_generator(latent_dim):
    img_shape = (32, 32, 3)
    generator = Sequential()

    generator.add(Dense(256, input_dim=latent_dim))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Dense(512))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Dense(1024))
    generator.add(LeakyReLU(alpha=0.2))
    generator.add(BatchNormalization(momentum=0.8))
    generator.add(Dense(np.prod(img_shape), activation='tanh'))
    generator.add(Reshape(img_shape))

    # model.summary()

    noise = Input(shape=(latent_dim,))
    img = generator(noise)

    return Model(noise, img)
