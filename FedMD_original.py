import numpy as np
from tensorflow.keras.models import clone_model, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping
# from keras.layers import Input
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt

from data_utils import generate_alignment_data
from Neural_Networks import remove_last_layer, build_generator, modify_set_weights


'''
    没有GAN
'''
class FedMD():
    def __init__(self, parties, public_dataset,
                 private_data, total_private_data,
                 private_test_data, N_alignment,
                 N_rounds,
                 N_logits_matching_round, logits_matching_batchsize,
                 N_private_training_round, private_training_batchsize):

        # self.parties = parties
        self.N_parties = len(parties)
        self.public_dataset = public_dataset
        self.private_data = private_data
        self.private_test_data = private_test_data
        self.N_alignment = N_alignment

        self.N_rounds = N_rounds
        self.N_logits_matching_round = N_logits_matching_round
        self.logits_matching_batchsize = logits_matching_batchsize
        self.N_private_training_round = N_private_training_round
        self.private_training_batchsize = private_training_batchsize

        self.collaborative_parties = []
        self.init_result = []

        losses = ['categorical_crossentropy', 'kullback_leibler_divergence']
        # Build the generator
        self.latent_dim = 100
        self.generator = build_generator(self.latent_dim)

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        validity = {}
        target_logits = {}

        print("start model initialization: ")
        for i in range(self.N_parties):
            print("model ", i)
            model_A_twin = None
            model_A_twin = clone_model(parties[i])
            model_A_twin.set_weights(parties[i].get_weights())
            model_A_twin.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                                 loss=['sparse_categorical_crossentropy', 'kullback_leibler_divergence'],
                                 metrics=["accuracy"])

            # 输入是 img， 输出是 label 和 valid
            private_data[i]["validity"] = np.ones((len(private_data[i]["X"]), 1))
            private_test_data["validity"] = np.ones((len(private_test_data["X"]), 1))

            print("start full stack training ... ")

            model_A_twin.fit(private_data[i]["X"], [private_data[i]["y"], private_data[i]["validity"]],
                             batch_size=32, epochs=25, shuffle=True, verbose=0,
                             validation_data=[private_test_data["X"],
                                              [private_test_data["y"], private_test_data["validity"]]],
                             callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)]
                             )

            print("full stack training done")

            model_A = remove_last_layer(model_A_twin, loss=losses)

            # # 判别器的输出
            target_logits[i], validity[i] = model_A(img)

            self.collaborative_parties.append({"model_logits": model_A,
                                               "model_classifier": model_A_twin,
                                               "model_weights": model_A_twin.get_weights()})

            print()
            del model_A, model_A_twin
        # END FOR LOOP

    def collaborative_training(self):
        # start collaborating training
        collaboration_performance = {i: [] for i in range(self.N_parties)}
        collaboration_loss = {i: [] for i in range(self.N_parties)}

        # Adversarial ground truths
        valid = np.ones((self.N_alignment, 1))
        fake = np.zeros((self.N_alignment, 1))

        r = 0
        while True:
            # At beginning of each round, generate new alignment dataset
            alignment_data = generate_alignment_data(self.public_dataset["X"],
                                                     self.public_dataset["y"],
                                                     self.N_alignment)
            # print("alignment_data shape:", alignment_data["X"].shape)

            print("round ", r)

            print("update logits ... ")
            # update logits
            logits = 0
            for d in self.collaborative_parties:
                d["model_logits"] = modify_set_weights(d["model_logits"], d["model_weights"])
                logits += d["model_logits"].predict(alignment_data["X"], verbose=0)[0]

            logits /= self.N_parties

            # test performance
            print("test performance ... ")

            for index, d in enumerate(self.collaborative_parties):
                y_pred = d["model_classifier"].predict(self.private_test_data["X"], verbose=0)[0].argmax(axis=1)
                collaboration_performance[index].append(np.mean(self.private_test_data["y"] == y_pred))
                collaboration_loss[index].append(d["model_classifier"].evaluate(self.private_test_data["X"],
                                                                                [self.private_test_data["y"],
                                                                                 np.ones((len(
                                                                                     self.private_test_data["X"]),
                                                                                          1))])[0])
                print(collaboration_performance[index][-1])
                print(collaboration_loss[index][-1])
                del y_pred

            r += 1
            if r > self.N_rounds:
                break

            print("updates models ...")
            for index, d in enumerate(self.collaborative_parties):
                # 训练判别器， 即每个client的local model
                print("model {0} starting alignment with public logits... ".format(index))

                weights_to_use = None
                weights_to_use = d["model_weights"]

                # logits (, 16)
                d["model_logits"] = modify_set_weights(d["model_logits"], weights_to_use)
                d["model_logits"].fit(alignment_data["X"], [logits, fake],
                                      batch_size=self.logits_matching_batchsize,
                                      epochs=self.N_logits_matching_round,
                                      shuffle=True, verbose=True)

                d["model_weights"] = d["model_logits"].get_weights()
                print("model {0} done alignment".format(index))

                print("model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                # private_logits =
                d["model_classifier"] = modify_set_weights(d["model_classifier"], weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"],
                                          [self.private_data[index]["y"],
                                           np.ones((len(self.private_data[index]["X"]), 1))],
                                          batch_size=self.private_training_batchsize,
                                          epochs=self.N_private_training_round,
                                          shuffle=True, verbose=True)

                d["model_weights"] = d["model_classifier"].get_weights()
                print("model {0} done private training. \n".format(index))
            # END FOR LOOP

        # END WHILE LOOP
        return collaboration_performance, collaboration_loss

    def sample_images(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        # sampled_labels = np.array([num for _ in range(r) for num in range(c)])
        gen_imgs = self.generator.predict(noise)
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        # print(gen_imgs.shape) # (100, 28, 28)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


