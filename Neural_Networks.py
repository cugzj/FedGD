import os
import errno
from tensorflow.keras.models import clone_model, load_model
from tensorflow.keras.layers import Dense, add, concatenate, Conv2D,Dropout,\
BatchNormalization, Flatten, MaxPooling2D, AveragePooling2D, Activation, Dropout, Reshape, LeakyReLU
# from keras.layers.advanced_activations import LeakyReLU
from tensorflow.keras import Sequential, Model, Input, backend
import tensorflow as tf


import numpy as np


def cnn_3layer_fc_model(n_classes,n1 = 128, n2=192, n3=256, dropout_rate = 0.2,input_shape = (28,28)):
    model_A, x = None, None
     
    x = Input(input_shape)
    if len(input_shape)==2: y = Reshape((input_shape[0], input_shape[1], 1))(x)

    else:
        y = Reshape((input_shape))(x)
    
    y = Conv2D(filters = n1, kernel_size = (3,3), strides = 1, padding = "same", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size = (2,2), strides = 1, padding = "same")(y)

    y = Conv2D(filters = n2, kernel_size = (2,2), strides = 2, padding = "valid", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Conv2D(filters = n3, kernel_size = (3,3), strides = 2, padding = "valid", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    #y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Flatten()(y)  # feature map
    logits = Dense(units = n_classes, activation = None, use_bias = False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-3))(y)  # 倒数第二层
    # valid = Dense(1, activation="sigmoid")(y)  # 判别输出
    # y = Activation("softmax")(logits)
    valid_logit = Dense(units=1, activation = None, use_bias = False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-3), name='dense-valid-1')(y)
    # 对于原始的完整模型而言，这些层不会被更新
    valid_logit.trainable = False
    y = Activation("softmax")(logits)
    valid = Activation("sigmoid")(valid_logit)

    model_A = Model(inputs = x, outputs = [y, valid])

    model_A.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-3), 
                        loss = "sparse_categorical_crossentropy",
                        metrics = ["accuracy"])
    return model_A
  
def cnn_2layer_fc_model(n_classes,n1 = 128, n2=256, dropout_rate = 0.2,input_shape = (28,28)):
    model_A, x = None, None
    
    x = Input(input_shape)
    if len(input_shape)==2:
        y = Reshape((input_shape[0], input_shape[1], 1))(x)
    else:
        y = Reshape((input_shape))(x)
    
    y = Conv2D(filters = n1, kernel_size = (3,3), strides = 1, padding = "same", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    y = AveragePooling2D(pool_size = (2,2), strides = 1, padding = "same")(y)


    y = Conv2D(filters = n2, kernel_size = (3,3), strides = 2, padding = "valid", 
            activation = None)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)
    y = Dropout(dropout_rate)(y)
    #y = AveragePooling2D(pool_size = (2,2), strides = 2, padding = "valid")(y)

    y = Flatten()(y)
    logits = Dense(units = n_classes, activation = None, use_bias = False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-3))(y)
    # valid = Dense(1, activation="sigmoid")(y)
    # y = Activation("softmax")(logits)
    valid_logit = Dense(units=1, activation = None, use_bias = False,
            kernel_regularizer=tf.keras.regularizers.l2(1e-3), name='dense-valid-1')(y)
    # 对于原始的完整模型而言，这些层不会被更新
    valid_logit.trainable = False
    y = Activation("softmax")(logits)
    valid = Activation("sigmoid")(valid_logit)
    # valid.trainable = False

    model_A = Model(inputs=x, outputs=[y, valid])

    model_A.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-3), 
                        loss = "sparse_categorical_crossentropy",
                        metrics = ["accuracy"])
    return model_A


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

    new_model = Model(inputs = model.inputs, outputs = [model.layers[-4].output, model.layers[-1].output])
    # new_model.set_weights(model.get_weights())
    """
        由于删除了softmax层，因此new_model的weights下标产生变化，这里手动实现set_weights的操作
    """
    # weights = model.get_weights()
    # new_model = modify_set_weights(new_model, weights)
    new_model.set_weights(model.get_weights())
    new_model.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-3),
                      loss = loss, loss_weights=[0.8,0.2])
    
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
    for pv, p, w in zip(param_values, new_model.weights, weights):
        if r == l-2:  # 手动调整最后两层的weight的赋值
            new_18 = w
            old_18 = p
            # layer_18 = pv
            # weight_value_tuples.append((w, p))
        elif r == l-1:
            new_19 = w
            old_19 = p
            # layer_19 = pv
            weight_value_tuples.append((old_18, new_19))
            weight_value_tuples.append((old_19, new_18))
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

# def reverse_set_weights(new_model, weights):
#     if len(new_model.weights) != len(weights):
#         raise ValueError('You called `set_weights(weights)` on layer "' +
#                          new_model.name + '" with a  weight list of length ' +
#                          str(len(weights)) + ', but the layer was expecting ' +
#                          str(len(new_model.weights)) + ' weights. Provided weights: ' +
#                          str(weights)[:50] + '...')
#     if not new_model.weights:
#         return
#     weight_value_tuples = []
#     param_values = backend.batch_get_value(new_model.weights)
#     r = 0
#     l = len(new_model.weights)
#     for pv, p, w in zip(param_values, new_model.weights, weights):
#         if r == l-2:  # 手动调整最后两层的weight的赋值
#             new_18 = w
#             old_18 = p
#             # layer_18 = pv
#             # weight_value_tuples.append((w, p))
#         elif r == l-1:
#             new_19 = w
#             old_19 = p
#             # layer_19 = pv
#             weight_value_tuples.append((old_18, new_19))
#             weight_value_tuples.append((old_19, new_18))
#             break
#         elif pv.shape != w.shape:
#             raise ValueError('Layer weight shape ' + str(pv.shape) +
#                              ' not compatible with '
#                              'provided weight shape ' + str(w.shape))
#         weight_value_tuples.append((p, w))
#         r += 1
#     backend.batch_set_value(weight_value_tuples)
#     return new_model


# build generator
# The generator takes noise as input and generates imgs
def build_generator(latent_dim):
    img_shape = (28,28)
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
