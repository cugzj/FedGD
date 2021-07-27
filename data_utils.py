import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.datasets import cifar10, cifar100, mnist, fashion_mnist
import scipy.io as sio
import tensorflow as tf
import pickle
import os


def load_MNIST_data(standarized = False, verbose = False):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    
    if standarized: 
        X_train = X_train/255
        X_test = X_test/255
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test -= mean_image
    
    if verbose == True: 
        print("MNIST dataset ... ")
        print("X_train shape :", X_train.shape)
        print("X_test shape :", X_test.shape)
        print("y_train shape :", y_train.shape)
        print("y_test shape :", y_test.shape)
    
    return X_train, y_train, X_test, y_test


def load_Fashion_MNIST_data(standarized=False, verbose=False):
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    if standarized:
        X_train = X_train / 255
        X_test = X_test / 255
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test -= mean_image

    if verbose == True:
        print("Fashion_MNIST dataset ... ")
        print("X_train shape :", X_train.shape)
        print("X_test shape :", X_test.shape)
        print("y_train shape :", y_train.shape)
        print("y_test shape :", y_test.shape)

    return X_train, y_train, X_test, y_test


def load_CIFAR_data(data_type="CIFAR10", label_mode="fine",
                    standarized=False, verbose=False):
    if data_type == "CIFAR10":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    elif data_type == "CIFAR100":
        (X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode=label_mode)
    else:
        print("Unknown Data type. Stopped!")
        return None

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    # substract mean and normalized to [-1/2,1/2]
    if standarized:
        X_train = X_train / 255
        X_test = X_test / 255
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test -= mean_image

    if verbose == True:
        print("X_train shape :", X_train.shape)
        print("X_test shape :", X_test.shape)
        print("y_train shape :", y_train.shape)
        print("y_test shape :", y_test.shape)

    return X_train, y_train, X_test, y_test


def load_CIFAR_from_local(local_dir, data_type="CIFAR10", with_coarse_label=False,
                          standarized=False, verbose=False):
    # dir_name = os.path.abspath(local_dir)
    if data_type == "CIFAR10":
        X_train, y_train = [], []
        for i in range(1, 6, 1):
            file_name = None
            file_name = os.path.join(local_dir + "data_batch_{0}".format(i))
            X_tmp, y_tmp = None, None
            with open(file_name, 'rb') as fo:
                datadict = pickle.load(fo, encoding='bytes')

            X_tmp = datadict[b'data']
            y_tmp = datadict[b'labels']
            X_tmp = X_tmp.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            y_tmp = np.array(y_tmp)

            X_train.append(X_tmp)
            y_train.append(y_tmp)
            del X_tmp, y_tmp
        X_train = np.vstack(X_train)
        y_train = np.hstack(y_train)

        file_name = None
        file_name = os.path.join(local_dir + "test_batch")
        with open(file_name, 'rb') as fo:
            datadict = pickle.load(fo, encoding='bytes')

            X_test = datadict[b'data']
            y_test = datadict[b'labels']
            X_test = X_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            y_test = np.array(y_test)


    elif data_type == "CIFAR100":
        file_name = None
        file_name = os.path.abspath(local_dir + "train")
        with open(file_name, 'rb') as fo:
            datadict = pickle.load(fo, encoding='bytes')
            X_train = datadict[b'data']
            if with_coarse_label:
                y_train = datadict[b'coarse_labels']
            else:
                y_train = datadict[b'fine_labels']
            X_train = X_train.reshape(50000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            y_train = np.array(y_train)

        file_name = None
        file_name = os.path.join(local_dir + "test")
        with open(file_name, 'rb') as fo:
            datadict = pickle.load(fo, encoding='bytes')
            X_test = datadict[b'data']
            if with_coarse_label:
                y_test = datadict[b'coarse_labels']
            else:
                y_test = datadict[b'fine_labels']
            X_test = X_test.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
            y_test = np.array(y_test)

    else:
        print("Unknown Data type. Stopped!")
        return None

    if standarized:
        X_train = X_train / 255
        X_test = X_test / 255
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test -= mean_image

    if verbose == True:
        print("X_train shape :", X_train.shape)
        print("X_test shape :", X_test.shape)
        print("y_train shape :", y_train.shape)
        print("y_test shape :", y_test.shape)

    return X_train, y_train, X_test, y_test


def load_EMNIST_data(file, verbose = False, standarized = False):
    """
    file should be the downloaded EMNIST file in .mat format.
    """    
    mat = sio.loadmat(file)
    data = mat["dataset"]
    
    X_train = data['train'][0,0]['images'][0,0]
    X_train = X_train.reshape((X_train.shape[0], 28, 28), order = "F")
    y_train = data['train'][0,0]['labels'][0,0]
    y_train = np.squeeze(y_train)
    y_train -= 1 #y_train is zero-based
    
    X_test = data['test'][0,0]['images'][0,0]
    X_test= X_test.reshape((X_test.shape[0], 28, 28), order = "F")
    y_test = data['test'][0,0]['labels'][0,0]
    y_test = np.squeeze(y_test)
    y_test -= 1 #y_test is zero-based
    
    if standarized: 
        X_train = X_train/255
        X_test = X_test/255
        mean_image = np.mean(X_train, axis=0)
        X_train -= mean_image
        X_test -= mean_image
    

    if verbose == True: 
        print("EMNIST-letter dataset ... ")
        print("X_train shape :", X_train.shape)
        print("X_test shape :", X_test.shape)
        print("y_train shape :", y_train.shape)
        print("y_test shape :", y_test.shape)
    
    return X_train, y_train, X_test, y_test


def generate_partial_data(X, y, class_in_use = "all", verbose = False):
    if class_in_use == "all":
        idx = np.ones_like(y, dtype = bool)
    else:
        idx = [y == i for i in class_in_use]
        idx = np.any(idx, axis = 0)
    # print("idx:", idx, len(idx))
    # print("X[idx]:", type(X), X[idx], len(X[idx]))
    X_incomplete, y_incomplete = X[idx], y[idx]
    if verbose == True:
        print("X shape :", X_incomplete.shape)
        print("y shape :", y_incomplete.shape)
    return X_incomplete, y_incomplete



def generate_bal_private_data(X, y, N_parties = 10, classes_in_use = range(11), 
                              N_samples_per_class = 20, data_overlap = False):
    """
    Input: 
    -- N_parties : int, number of collaboraters in this activity;
    -- classes_in_use: array or generator, the classes of EMNIST-letters dataset 
    (0 <= y <= 25) to be used as private data; 
    -- N_sample_per_class: int, the number of private data points of each class for each party
    
    return: 
    
    """
    priv_data = [None] * N_parties
    combined_idx = np.array([], dtype = np.int16)
    for cls in classes_in_use:
        idx = np.where(y == cls)[0]
        idx = np.random.choice(idx, N_samples_per_class * N_parties, 
                               replace = data_overlap)
        combined_idx = np.r_[combined_idx, idx]
        for i in range(N_parties):           
            idx_tmp = idx[i * N_samples_per_class : (i + 1)*N_samples_per_class]
            if priv_data[i] is None:
                tmp = {}
                tmp["X"] = X[idx_tmp]
                tmp["y"] = y[idx_tmp]
                tmp["idx"] = idx_tmp
                priv_data[i] = tmp
            else:
                priv_data[i]['idx'] = np.r_[priv_data[i]["idx"], idx_tmp]
                priv_data[i]["X"] = np.vstack([priv_data[i]["X"], X[idx_tmp]])
                priv_data[i]["y"] = np.r_[priv_data[i]["y"], y[idx_tmp]]
                
                
    total_priv_data = {}
    total_priv_data["idx"] = combined_idx
    total_priv_data["X"] = X[combined_idx]
    total_priv_data["y"] = y[combined_idx]
    print(len(combined_idx))
    return priv_data, total_priv_data


def generate_alignment_data(X, y, N_alignment = 3000):
    
    split = StratifiedShuffleSplit(n_splits=1, train_size= N_alignment)
    if N_alignment == "all":
        alignment_data = {}
        alignment_data["idx"] = np.arange(y.shape[0])
        alignment_data["X"] = X
        alignment_data["y"] = y
        return alignment_data
    for train_index, _ in split.split(X, y):
        X_alignment = X[train_index]
        y_alignment = y[train_index]
    alignment_data = {}
    alignment_data["idx"] = train_index
    alignment_data["X"] = X_alignment
    alignment_data["y"] = y_alignment
    
    return alignment_data

def FL_loss(target, output, from_logits=False):
    """
    output = [y, validity]
    total loss contains two parts:
        part 1: binary crossentropy loss between an output tensor and a target tensor
        part 2: KL divergence loss between an output distribution and a target distribution

    # returns
        A tensor
    """

    # Note: tf.nn.sigmoid_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.

    # if not from_logits:
    #     # transform back to logits
    #     _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
    #     output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    #     output = tf.log(output / (1 - output))

    loss_1 = tf.nn.sigmoid_cross_entropy_with_logits(labels=target[0], logits=output[0])
    loss_2 = tf.keras.losses.KLDivergence()(target[1], output[1])

    return loss_1 + loss_2


def generate_imbal_CIFAR_private_data(X, y, y_super, classes_per_party, N_parties,
                                      samples_per_class=7):
    priv_data = [None] * N_parties
    combined_idxs = []
    count = 0
    for subcls_list in classes_per_party:
        idxs_per_party = []
        for c in subcls_list:
            idxs = np.flatnonzero(y == c)
            idxs = np.random.choice(idxs, samples_per_class, replace=False)
            idxs_per_party.append(idxs)
        idxs_per_party = np.hstack(idxs_per_party)
        combined_idxs.append(idxs_per_party)

        dict_to_add = {}
        dict_to_add["idx"] = idxs_per_party
        dict_to_add["X"] = X[idxs_per_party]
        # dict_to_add["y"] = y[idxs_per_party]
        # dict_to_add["y_super"] = y_super[idxs_per_party]
        dict_to_add["y"] = y_super[idxs_per_party]
        priv_data[count] = dict_to_add
        count += 1

    combined_idxs = np.hstack(combined_idxs)
    total_priv_data = {}
    total_priv_data["idx"] = combined_idxs
    total_priv_data["X"] = X[combined_idxs]
    # total_priv_data["y"] = y[combined_idxs]
    # total_priv_data["y_super"] = y_super[combined_idxs]
    total_priv_data["y"] = y_super[combined_idxs]
    return priv_data, total_priv_data

