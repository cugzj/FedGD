import numpy as np
from tensorflow.keras.models import clone_model, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping
# from keras.layers import Input
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from data_utils import generate_alignment_data
from ResNet import build_generator
from model import remove_last_layer


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

        losses = ['categorical_crossentropy']
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
                                 loss=['sparse_categorical_crossentropy'],
                                 metrics=["accuracy"])

            # 输入是 img， 输出是 label 和 valid
            private_data[i]["validity"] = np.ones((len(private_data[i]["X"]), 1))
            private_test_data["validity"] = np.ones((len(private_test_data["X"]), 1))

            print("start full stack training ... ")

            # model_A_twin.fit(private_data[i]["X"], private_data[i]["y"],
            #                  batch_size=32, epochs=25, shuffle=True, verbose=0,
            #                  validation_data=[private_test_data["X"],private_test_data["y"]],
            #                  callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)]
            #                  )

            print("full stack training done")

            model_A = remove_last_layer(model_A_twin, loss=self.mutual_KL_loss)
            # layer_valid_logigts = model_A.get_layer(index=len(model_A.layers)-3)
            # layer_valid_logigts.tainable = True

            # # 判别器的输出
            target_logits[i] = model_A(img)

            self.collaborative_parties.append({"model_logits": model_A,
                                               "model_classifier": model_A_twin,
                                               "model_weights": model_A_twin.get_weights()})

            print()
            del model_A, model_A_twin
        # END FOR LOOP
        # validity_output = np.sum(validity[i] for i in range(self.N_parties)) / self.N_parties
        logits_output = np.sum(target_logits[i] for i in range(self.N_parties)) / self.N_parties
        self.combined = Model(z, logits_output, name='combined')
        self.combined.compile(loss=self.mutual_combined_loss, optimizer=Adam(0.0002, 0.5))

    def generate_training_data_for_gan(self):
        alignment_data = {}
        for i in range(self.N_parties):
            alignment_data[i] = generate_alignment_data(self.private_data[i]["X"],
                                                        self.private_data[i]["y"],
                                                        self.N_alignment)
        return alignment_data

    def mutual_KL_loss(self, logit_target, logit_pre):
        """
            The mutual information metric we aim to minimize
            logit_true = [logit_avg, logit_true]
        """
        # print(logit_target.shape, logit_target[0].shape, logit_target[0][0].shape, logit_target[0][1].shape, logit_pre)
        loss_categorical_crossentropy, loss_KL = 0, 0
        for i in range(self.N_alignment):
            loss_categorical_crossentropy = K.categorical_crossentropy(logit_target[i][0], logit_pre)
            # entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))
            KL = tf.keras.losses.KLDivergence()
            loss_KL = KL(logit_target[i][1], logit_pre)
        # loss_KL = K.sum(logit_target[1])
        return loss_categorical_crossentropy+loss_KL
        # return loss_categorical_crossentropy

    def mutual_combined_loss(self, logit_target, logit_output):
        """
            The mutual information metric we aim to minimize
            logit_pre = [logit_pre[1], logit_pre[2], ..., logit_pre[K]]
            logit_target = [logit_target[1], logit_target[2], ..., logit_target[K]]
        """
        KL = tf.keras.losses.KLDivergence()
        loss_KL = 0
        for index in range(self.N_alignment):
            for iidex in range(self.N_parties):
                loss_KL += KL(logit_target[index][iidex], logit_output[index])
        # for index in range(self.N_parties):
        #     loss_KL += KL(logit_target[index], logit_output[index])
        return loss_KL

    def collaborative_training(self):
        # start collaborating training
        collaboration_performance = {i: [] for i in range(self.N_parties)}
        collaboration_loss = {i: [] for i in range(self.N_parties)}

        # Adversarial ground truths
        # valid = np.ones((self.N_alignment, 1))
        # fake = np.zeros((self.N_alignment, 1))

        r = 0
        record_generator_result = []
        while True:
            # At beginning of each round, generate new alignment dataset
            # alignment_data = generate_alignment_data(self.public_dataset["X"],
            #                                          self.public_dataset["y"],
            #                                          self.N_alignment)
            alignment_data = self.generate_training_data_for_gan()
            # print("alignment_data shape:", alignment_data["X"].shape)
            noise = np.random.normal(0, 1, (self.N_alignment, self.latent_dim))

            # The labels of the digits that the generator tries to create an
            # image representation of
            # sampled_labels = np.random.randint(0, 10, (self.N_alignment, 1))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)
            print("round ", r)

            logits_target_combined = []
            print("update logits ... ")
            # update logits
            logits = 0
            for d in self.collaborative_parties:
                d["model_logits"].set_weights(d["model_weights"])
                logits = logits + d["model_logits"].predict(gen_imgs, verbose=0)

            logits /= self.N_parties

            # test performance
            print("test performance ... ")

            # for index, d in enumerate(self.collaborative_parties):
            #     y_pred = d["model_classifier"].predict(self.private_test_data["X"], verbose=0).argmax(axis=1)
            #     collaboration_performance[index].append(np.mean(self.private_test_data["y"] == y_pred))
            #     collaboration_loss[index].append(d["model_classifier"].evaluate(self.private_test_data["X"],
            #                                                                     self.private_test_data["y"])[0])
            #     print(collaboration_performance[index][-1])
            #     print(collaboration_loss[index][-1])
            #     del y_pred

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
                d["model_logits"].set_weights(weights_to_use)
                logits_true = d["model_logits"].predict(alignment_data[index]["X"], verbose=0)
                logits = np.asarray(logits).astype(np.float32)
                logits_target, logits_target_t = [], []
                for i in range(len(logits_true)):
                    logits_target.append([logits[i], logits_true[i]])
                    logits_target_t.append([logits_true[i], logits_true[i]])
                # print(type(np.asarray(logits_target)))
                d["model_logits"].fit(gen_imgs, [logits_target],
                                      batch_size=self.logits_matching_batchsize,
                                      epochs=self.N_logits_matching_round,
                                      shuffle=True, verbose=True)

                d["model_logits"].fit(alignment_data[index]["X"], [logits_target_t],
                                      batch_size=self.logits_matching_batchsize,
                                      epochs=self.N_logits_matching_round,
                                      shuffle=True, verbose=True)

                d["model_weights"] = d["model_logits"].get_weights()
                print("model {0} done alignment".format(index))

                print("model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                # private_logits =
                d["model_classifier"].set_weights(weights_to_use)
                d["model_classifier"].fit(self.private_data[index]["X"],
                                          self.private_data[index]["y"],
                                          batch_size=self.private_training_batchsize,
                                          epochs=self.N_private_training_round,
                                          shuffle=True, verbose=True)

                d["model_weights"] = d["model_classifier"].get_weights()
                print("model {0} done private training. \n".format(index))

                logits_target_combined.append(logits_true)  # 当前party的true data的logits输出, shape = (N_alignment, 16)
            # END FOR LOOP

            # ---------------------
            #  Train Generator
            # ---------------------
            target_combined = []
            for i in range(self.N_alignment):
                temp = []
                for j in range(len(logits_target_combined)):
                    temp.append(logits_target_combined[j][i])
                target_combined.append(temp)
            # Train the generator
            # g_loss = self.combined.train_on_batch(noise, [logits, valid])
            print("generator starting training with noise... ")
            self.combined.fit(noise, [target_combined],
                              batch_size=self.logits_matching_batchsize,
                              epochs=self.N_logits_matching_round,
                              shuffle=True, verbose=True)
            record_generator_result.append(self.combined.history.history["loss"][-1])

        # END WHILE LOOP
        return collaboration_performance, collaboration_loss, record_generator_result

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


