import os
import errno
import argparse
import sys
import pickle

import numpy as np
from tensorflow.keras.models import load_model, clone_model
from data_utils import load_MNIST_data, load_Fashion_MNIST_data, \
    load_CIFAR_data, load_EMNIST_data, generate_bal_private_data, generate_partial_data
from FedMD import FedMD
# from FedMD_original import FedMD
# from FedMD_gan import FedMD
# from FedMD_logits import FedMD
from utility import plot_history, show_performance
from Neural_Networks import train_models, cnn_2layer_fc_model, cnn_3layer_fc_model


def parseArg():
    parser = argparse.ArgumentParser(description='FedMD, a federated learning framework. \
    Participants are training collaboratively. ')
    parser.add_argument('-conf', metavar='conf_file', nargs=1, 
                        help='the config file for FedMD.'
                       )

    conf_file = os.path.abspath("conf/EMNIST_balance_10_conf.json")
    
    if len(sys.argv) > 1:
        args = parser.parse_args(sys.argv[1:])
        if args.conf:
            conf_file = args.conf[0]
    return conf_file


CANDIDATE_MODELS = {"2_layer_CNN": cnn_2layer_fc_model,
                    "3_layer_CNN": cnn_3layer_fc_model}

if __name__ == "__main__":
    conf_file =  parseArg()
    with open(conf_file, "r") as f:
        conf_dict = eval(f.read())
        
        emnist_data_dir = conf_dict["EMNIST_dir"]    
        N_parties = conf_dict["N_parties"]
        Public_dataset = conf_dict["public_dataset"]
        private_classes = conf_dict["private_classes"]
        public_classes = conf_dict["public_classes"]
        N_samples_per_class = conf_dict["N_samples_per_class"]

        # public_dataset = conf_dict["public_dataset"]
        # n_classes = conf_dict["n_classes"]
        model_config = conf_dict["models"]
        train_params = conf_dict["pre_train_params"]
        save_names = conf_dict["model_saved_names"]
        early_stopping = conf_dict["early_stopping"]
        alg = conf_dict["algorithm"]
        n_classes = len(public_classes) + len(private_classes)
        
        N_rounds = conf_dict["N_rounds"]
        N_alignment = conf_dict["N_alignment"]
        N_private_training_round = conf_dict["N_private_training_round"]
        private_training_batchsize = conf_dict["private_training_batchsize"]
        N_logits_matching_round = conf_dict["N_logits_matching_round"]
        logits_matching_batchsize = conf_dict["logits_matching_batchsize"]
        model_saved_dir = conf_dict["model_saved_dir"]  # whether exists a pre-trained model (on public dataset)
        
        result_save_dir = conf_dict["result_save_dir"]

    
    del conf_dict, conf_file

    if Public_dataset == "MNIST":
        # load MNIST
        X_train_MNIST, y_train_MNIST, X_test_MNIST, y_test_MNIST \
        = load_MNIST_data(standarized = True, verbose = True)

        public_dataset = {"X": X_train_MNIST, "y": y_train_MNIST}
        public_test_dataset = {"X": X_test_MNIST, "y": y_test_MNIST}
        del  X_train_MNIST, y_train_MNIST, X_test_MNIST, y_test_MNIST

    elif Public_dataset == "Fashion_MNIST":
        # load Fashion_MNIST
        X_train_Fashion_MNIST, y_train_Fashion_MNIST, X_test_Fashion_MNIST, y_test_Fashion_MNIST \
            = load_Fashion_MNIST_data(standarized=True, verbose=True)

        public_dataset = {"X": X_train_Fashion_MNIST, "y": y_train_Fashion_MNIST}
        public_test_dataset = {"X": X_test_Fashion_MNIST, "y": y_test_Fashion_MNIST}
        # print("public ")
        del X_train_Fashion_MNIST, y_train_Fashion_MNIST, X_test_Fashion_MNIST, y_test_Fashion_MNIST
    else:
        # load E_MNIST
        X_train_EMNIST, y_train_EMNIST, X_test_EMNIST, y_test_EMNIST \
            = load_EMNIST_data(emnist_data_dir,
                               standarized=True, verbose=True)

        X_p, y_p = generate_partial_data(X=X_train_EMNIST, y=y_train_EMNIST,
                                         class_in_use=private_classes, verbose=True)
        X_test_p, y_test_p = generate_partial_data(X=X_test_EMNIST, y=y_test_EMNIST,
                                                   class_in_use=private_classes, verbose=True)
        public_dataset = {"X": X_p, "y": y_p}
        public_test_dataset = {"X": X_test_p, "y": y_test_p}
        # print("public ")
        del X_train_EMNIST, y_train_EMNIST, X_test_EMNIST, y_test_EMNIST

    X_train_EMNIST, y_train_EMNIST, X_test_EMNIST, y_test_EMNIST \
    = load_EMNIST_data(emnist_data_dir,
                       standarized = True, verbose = True)

    # generate private data
    private_data, total_private_data \
    = generate_bal_private_data(X_train_EMNIST, y_train_EMNIST, 
                            N_parties = N_parties,                                          
                            classes_in_use = private_classes, 
                            N_samples_per_class = N_samples_per_class, 
                            data_overlap = False)
    
    X_tmp, y_tmp = generate_partial_data(X = X_test_EMNIST, y= y_test_EMNIST, 
                                         class_in_use = private_classes, verbose = True)
    private_test_data = {"X": X_tmp, "y": y_tmp}
    del X_tmp, y_tmp


    parties = []
    if model_saved_dir is None:
        # pretrain_models = []
        for i, item in enumerate(model_config):
            model_name = item["model_type"]
            model_params = item["params"]
            tmp = CANDIDATE_MODELS[model_name](n_classes=n_classes,
                                         input_shape=(28, 28),
                                         **model_params)

            print("model {0} : {1}".format(i, save_names[i]))
            print(tmp.summary())
            parties.append(tmp)

            del model_name, model_params, tmp

        record_result = train_models(parties,
                                     public_dataset["X"],
                                     public_dataset["y"],
                                     public_test_dataset["X"],
                                     public_test_dataset["y"],
                                     save_dir=model_saved_dir, save_names=save_names,
                                     early_stopping=early_stopping,
                                     **train_params
                                     )

        with open('pretrain_result.pkl', 'wb') as f:
            pickle.dump(record_result, f, protocol=pickle.HIGHEST_PROTOCOL)

    else:
        dpath = os.path.abspath(model_saved_dir)
        model_names = os.listdir(dpath)
        for name in model_names:
            tmp = None
            tmp = load_model(os.path.join(dpath ,name))
            parties.append(tmp)

    fedmd = FedMD(parties,
                  public_dataset = public_dataset,
                  private_data = private_data, 
                  total_private_data = total_private_data,
                  private_test_data = private_test_data,
                  N_rounds = N_rounds,
                  N_alignment = N_alignment, 
                  N_logits_matching_round = N_logits_matching_round,
                  logits_matching_batchsize = logits_matching_batchsize, 
                  N_private_training_round = N_private_training_round, 
                  private_training_batchsize = private_training_batchsize)
    
    initialization_result = fedmd.init_result
    # pooled_train_result = fedmd.pooled_train_result
    
    collaboration_performance, collaboration_loss, record_generator_result = fedmd.collaborative_training()
    # collaboration_performance, collaboration_loss = fedmd.collaborative_training()
    print("collaboration_performance:", collaboration_performance)
    print("collaboration_loss:", collaboration_loss)
    # print("record_generator_result:", record_generator_result)
    
    if result_save_dir is not None:
        save_dir_path = os.path.abspath(result_save_dir)
        #make dir
        try:
            os.makedirs(save_dir_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise    

    with open(os.path.join(save_dir_path, 'init_result.pkl'), 'wb') as f:
        pickle.dump(initialization_result, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(save_dir_path, 'pooled_train_result.pkl'), 'wb') as f:
    #     pickle.dump(pooled_train_result, f, protocol=pickle.HIGHEST_PROTOCOL)
    # col_performance = 'col_performance_without_gan_mnist' + str(N_rounds) + '.pkl'
    with open(os.path.join(save_dir_path, 'col_performance_'+str(alg)+'_'+str(Public_dataset)+'_'+str(N_rounds)+'_'+str(N_alignment)+'_'+str(N_samples_per_class*6)+'.pkl'), 'wb') as f:
        pickle.dump(collaboration_performance, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(save_dir_path, 'col_loss.pkl'), 'wb') as f:
        pickle.dump(collaboration_loss, f, protocol=pickle.HIGHEST_PROTOCOL)
    # with open(os.path.join(save_dir_path, 'generator_loss_'+str(N_rounds)+'.pkl'), 'wb') as f:
    #     pickle.dump(record_generator_result, f, protocol=pickle.HIGHEST_PROTOCOL)

