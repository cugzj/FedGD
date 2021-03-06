import numpy as np
from tensorflow.keras.models import clone_model, load_model, Model
from tensorflow.keras.callbacks import EarlyStopping
# from keras.layers import Input
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
from functools import partial
import tensorflow.keras.backend as K


from data_utils import generate_alignment_data
from ResNet_v2 import remove_last_layer, build_generator, modify_set_weights, modify_set_weights_fro_classifier

tf.compat.v1.enable_eager_execution()

class FedGD():
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

        # losses = ["categorical_crossentropy", 'binary_crossentropy']
        # Build the generator
        self.latent_dim = 100
        self.generator = build_generator(self.latent_dim)

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        validity = {}
        target_logits = {}
        D_kl = {}
        
        print("start model initialization: ")
        for i in range(self.N_parties):
            print("model ", i)
            model_A_twin = None
            model_A_twin = clone_model(parties[i])
            model_A_twin.set_weights(parties[i].get_weights())
            model_A_twin.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                                 loss = ['sparse_categorical_crossentropy', 'binary_crossentropy'],
                                 metrics=["accuracy"])

            # ????????? img??? ????????? label ??? valid
            private_data[i]["validity"] = np.ones((len(private_data[i]["X"]), 1))
            private_test_data["validity"] = np.ones((len(private_test_data["X"]), 1))

            print("start full stack training ... ")

            # ?????????label?????????
            model_A_twin.fit(private_data[i]["X"], [private_data[i]["y"], private_data[i]["validity"]],
                             batch_size = 32, epochs = 25, shuffle=True, verbose = 0,
                             validation_data = [private_test_data["X"], [private_test_data["y"], private_test_data["validity"]]],
                             callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5)]
                            )

            print("full stack training done")
            
            model_A = remove_last_layer(model_A_twin,
                                        loss=[self.mutual_KL_loss, 'binary_crossentropy'])
            # ???valid?????????????????????????????????
            layer_valid_logigts = model_A.get_layer(index=len(model_A.layers)-3)
            layer_valid_logigts.tainable = True

            # model_A.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3), loss=losses)
            # # ??????????????????
            target_logits[i], validity[i] = model_A(img)

            self.collaborative_parties.append({"model_logits": model_A,
                                               "model_classifier": model_A_twin,
                                               "model_weights": model_A_twin.get_weights()})
            # print(model_A_twin.history.history)
            # self.init_result.append({"val_acc": model_A_twin.history.history['val_activation_30_acc'],
            #                          "train_acc": model_A_twin.history.history['activation_30_acc'],
            #                          "val_loss": model_A_twin.history.history['val_activation_30_loss'],
            #                          "train_loss": model_A_twin.history.history['activation_30_loss'],
            #                         })
            # self.init_result.append(model_A_twin.history.history)

            # D_kl[i] = np.sum(img * np.log(img / self.private_data[i]["X"]))
            # D_kl[i] = tf.keras.losses.KLDivergence()(img, self.private_data[i]["X"])
            # D_kl[i] = tf.reduce_sum(tf.multiply(img, tf.math.log(tf.divide(img, self.private_data[i]["X"]))))

            print()
            del model_A, model_A_twin
        #END FOR LOOP
        
        # print("calculate the theoretical upper bounds for participants: ")
        #
        # self.upper_bounds = []
        # self.pooled_train_result = []
        # for model in parties:
        #     model_ub = clone_model(model)
        #     model_ub.set_weights(model.get_weights())
        #     model_ub.compile(optimizer=tf.keras.optimizers.Adam(lr = 1e-3),
        #                      loss = "sparse_categorical_crossentropy",
        #                      metrics = ["acc"])
        #
        #     model_ub.fit(total_private_data["X"], total_private_data["y"],
        #                  batch_size = 32, epochs = 50, shuffle=True, verbose = 1,
        #                  validation_data = [private_test_data["X"], private_test_data["y"]],
        #                  callbacks=[EarlyStopping(monitor='val_acc', min_delta=0.001, patience=5)])
        #
        #     self.upper_bounds.append(model_ub.history.history["val_acc"][-1])
        #     self.pooled_train_result.append({"val_acc": model_ub.history.history["val_acc"],
        #                                      "acc": model_ub.history.history["acc"]})
        #
        #     del model_ub
        # print("the upper bounds are:", self.upper_bounds)
        # print('D_kl', type(list(D_kl.values())[0]), list(D_kl.values())[0].numpy())
        # print(sorted(list(D_kl.values())))
        # max_distance = sorted(D_kl.items(), key=lambda k: k[1])[-1][1]
        # combined_output = []
        # for i in range(len(target_logits)):
        #     combined_output.append(target_logits[i])
        # target_logits_output = np.sum(target_logits[i] for i in range(self.N_parties)) / self.N_parties
        validity_output = np.sum(validity[i] for i in range(self.N_parties)) / self.N_parties
        logits_output = np.sum(target_logits[i] for i in range(self.N_parties)) / self.N_parties
        # logits_output = []
        # for i in range(self.N_parties):
        #     logits_output.append(target_logits[i])
        # self.combined = Model(z, [target_logits_output, validity_output, max_distance], name='combined')
        self.combined = Model(z, [logits_output, validity_output], name='combined')
        self.combined.compile(loss=[self.mutual_combined_loss, 'binary_crossentropy'], loss_weights=[0.8,0.2],
                              optimizer=Adam(0.0002, 0.5))

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

    def generate_training_data_for_gan(self):
        alignment_data = {}
        for i in range(self.N_parties):
            alignment_data[i] = generate_alignment_data(self.private_data[i]["X"],
                                                        self.private_data[i]["y"],
                                                        self.N_alignment)
        return alignment_data

    def collaborative_training(self):
        # start collaborating training
        collaboration_performance = {i: [] for i in range(self.N_parties)}
        collaboration_loss = {i: [] for i in range(self.N_parties)}

        # Adversarial ground truths
        valid = np.ones((self.N_alignment, 1))
        fake = np.zeros((self.N_alignment, 1))

        r = 0
        record_generator_result = []
        while True:
            # At beginning of each round, generate new alignment dataset
            alignment_data = self.generate_training_data_for_gan()
            # alignment_data = generate_alignment_data(self.)
            # print("alignment_data shape:", alignment_data["X"].shape)
            # Sample noise as generator input
            noise = np.random.normal(0, 1, (self.N_alignment, self.latent_dim))

            # The labels of the digits that the generator tries to create an
            # image representation of
            # sampled_labels = np.random.randint(0, 10, (self.N_alignment, 1))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)

            print("round ", r)

            print("update logits ... ")
            # update logits
            logits = 0
            actual_logits = 0
            for d in self.collaborative_parties:
                d["model_logits"] = modify_set_weights(d["model_logits"], d["model_weights"])
                # d["model_logits"].set_weights(d["model_weights"])
                # print(d["model_logits"].predict(alignment_data["X"], verbose = 0), len(d["model_logits"].predict(alignment_data["X"], verbose = 0)))
                # logits += d["model_logits"].predict(gen_imgs, verbose=0)[0]
                logits += d["model_logits"].predict(gen_imgs, verbose=0)[0]
                # actual_logits += d["model_logits"].predict(alignment_data["X"], verbose=0)[0]

            logits /= self.N_parties
            # actual_logits /= self.N_parties

            # test performance
            print("test performance ... ")

            for index, d in enumerate(self.collaborative_parties):
                y_pred = d["model_classifier"].predict(self.private_test_data["X"], verbose = 0)[0].argmax(axis = 1)
                collaboration_performance[index].append(np.mean(self.private_test_data["y"] == y_pred))
                collaboration_loss[index].append(d["model_classifier"].evaluate(self.private_test_data["X"],
                                                                                [self.private_test_data["y"],
                                                                                 np.ones((len(self.private_test_data["X"]), 1))])[0])
                print(collaboration_performance[index][-1])
                print(collaboration_loss[index][-1])
                del y_pred

            r+= 1
            if r > self.N_rounds:
                break

            # distance = {}
            logits_target_combined = []
            print("updates models ...")
            for index, d in enumerate(self.collaborative_parties):
                # ?????????????????? ?????????client???local model
                print("model {0} starting alignment with public logits... ".format(index))

                weights_to_use = None
                weights_to_use = d["model_weights"]

                # logits (, 16)
                d["model_logits"] = modify_set_weights(d["model_logits"], weights_to_use)
                # print(d["model_logits"].get_layer(index=len(d["model_logits"].layers)-3))
                logits_true = d["model_logits"].predict(alignment_data[index]["X"], verbose=0)[0]
                logits_target, logits_target_t = [], []
                for i in range(len(logits_true)):
                    logits_target.append([logits[i], logits_true[i]])
                    logits_target_t.append([logits_true[i], logits_true[i]])
                d["model_logits"].fit(gen_imgs, [logits_target, fake],
                                      batch_size = self.logits_matching_batchsize,
                                      epochs = self.N_logits_matching_round,
                                      shuffle=True, verbose = True)
                d["model_logits"].fit(alignment_data[index]["X"], [logits_target_t, valid],
                                      batch_size=self.logits_matching_batchsize,
                                      epochs=self.N_logits_matching_round,
                                      shuffle=True, verbose=True)
                # d["model_logits"].train_on_batch(gen_imgs, [logits, fake])

                d["model_weights"] = d["model_logits"].get_weights()
                print("model {0} done alignment".format(index))

                print("model {0} starting training with private data... ".format(index))
                weights_to_use = None
                weights_to_use = d["model_weights"]
                # private_logits =
                d["model_classifier"] = modify_set_weights_fro_classifier(d["model_classifier"], weights_to_use)
                # print(d["model_classifier"].get_layer(index=len(d["model_classifier"].layers)-2))
                d["model_classifier"].fit(self.private_data[index]["X"],
                                          [self.private_data[index]["y"], np.ones((len(self.private_data[index]["X"]), 1))],
                                          batch_size = self.private_training_batchsize,
                                          epochs = self.N_private_training_round,
                                          shuffle=True, verbose = True)

                d["model_weights"] = d["model_classifier"].get_weights()
                print("model {0} done private training. \n".format(index))

                logits_target_combined.append(logits_true)  # ??????party???true data???logits??????, shape = (N_alignment, 16)

                # ---------------------
                #  Calculate distance between gen_imgs and private_data
                # ---------------------
                # distance[index] = np.sum(gen_imgs*np.log(gen_imgs/self.private_data["X"]))

            # max_d = sorted(distance.items(), key=lambda kv: (kv[1], kv[0]))[-1][1]
            #END FOR LOOP

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
            self.combined.fit(noise, [target_combined, valid],
                              batch_size=self.logits_matching_batchsize,
                              epochs=self.N_logits_matching_round,
                              shuffle=True, verbose=True)
            record_generator_result.append(self.combined.history.history["loss"][-1])
            # print ("%d [G loss: %f]" % (r, g_loss[0]))
            # If at save interval => save generated image samples
            if r % 2 == 0:
                self.sample_images(r)

        #END WHILE LOOP
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
                axs[i,j].imshow(gen_imgs[cnt,:,:], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()


