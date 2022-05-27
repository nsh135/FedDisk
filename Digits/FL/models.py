"""Models used for Federated Learning"""

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Activation
from tensorflow.keras.layers import BatchNormalization
import time, os
##for prepare_data
import pickle
from skimage.transform import resize
import numpy as np
# for saving image (testing in main)
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.regularizers import l2



tf.config.run_functions_eagerly(True)

toGrey = True
if toGrey:
    digit_input_shape = (28,28,1)
else:
    digit_input_shape = (28,28,3)
digit_num_classes = 10

def prepare_data(dataset):
    """
    dataset names: MNIST, MNIST_M, SVHN, SynthDigits, USPS
    """
    for p in range(10):   
        file = open('../Data/{}/partitions/train_part{}.pkl'.format(dataset, p), 'rb')
        X, y = pickle.load(file)
        file.close()
        if p==0: # first partition
            X_train=X
            y_train=y
        else:
            X_train = np.concatenate((X_train, X), axis=0)
            y_train = np.concatenate((y_train, y), axis=0)
    #load testing file 
    file = open('../Data/{}/test.pkl'.format(dataset), 'rb')
    X_test, y_test = pickle.load(file)
    file.close()
    
    #resize images and scale (/255)
    X_train = np.float32(resize(X_train, (X_train.shape[0],28,28,3)))
    X_test = np.float32(resize(X_test, (X_test.shape[0],28,28,3)))
    
    print("Loading Dataset: {}".format(dataset))
    print("X_train.shape: {}".format( X_train.shape) )
    print("y_train.shape: {}".format( y_train.shape) )
    print("X_test.shape: {}".format(X_test.shape))
    print("y_test.shape: {}".format( y_test.shape))
    return X_train, y_train, X_test, y_test

class benchmark_models():
    """ Bench mark model for Digits """
    def digit_model_fedbn():
        """ Batch Normalization"""
        model = Sequential()
        model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same', input_shape=digit_input_shape))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(18, (3, 3), kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(32, kernel_initializer='he_uniform'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(digit_num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
        return model

    def digit_model_fedavg():
        model = Sequential()
        model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same', input_shape=digit_input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(14, (3, 3), kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(32, kernel_initializer='he_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(digit_num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
        return model

    def digit_model_fedDisk():
        """ Model for distribution skewed correction"""
        model = Sequential()
        model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same', input_shape=digit_input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Conv2D(22, (3, 3), kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(32, kernel_initializer='he_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(digit_num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
        return model

    def shallow_model(num_neuron):
        """ shallow model to determine sample weights"""
        model = Sequential()
        model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same', input_shape=digit_input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3), kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(num_neuron, kernel_initializer='he_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(2))
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=opt, loss=bce, metrics=['accuracy'])
        return model
    def shallow_model_1d(num_neuron):
        """ shallow model to determine sample weights"""
        model = Sequential()
        model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform', padding='same', input_shape=digit_input_shape))
        model.add(Activation('relu'))
        model.add(Conv2D(16, (3, 3), kernel_initializer='he_uniform', padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(MaxPooling2D((2, 2)))
        model.add(Flatten())
        model.add(Dense(num_neuron, kernel_initializer='he_uniform'))
        model.add(Activation('relu'))
        model.add(Dense(2))
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        opt = tf.keras.optimizers.Adam(learning_rate=0.01)
        model.compile(optimizer=opt, loss=bce, metrics=['accuracy'])
        return model

    def fnn_model(num_neuron, input_shape):
        """ shallow model to determine sample weights"""
        model = Sequential()
        model.add(Dense(32, kernel_initializer='he_uniform',input_shape=input_shape, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(Activation('relu'))
        model.add(Dense(num_neuron, kernel_initializer='he_uniform', kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
        model.add(Activation('relu'))
        model.add(Dense(2, activation='sigmoid',kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01) ))
        model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])
        return model

class CVAE(tf.keras.Model): 
    """Convolutional variational autoencoder."""
    def __init__(self, latent_dim):
        super(CVAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=digit_input_shape),
                tf.keras.layers.Conv2D(
                    filters=16, kernel_size=(3,3), strides=(2, 2), activation='relu'),
                # tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=32, kernel_size=(3,3), strides=(2, 2), activation='relu'),
                # tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Conv2D(
                    filters=64, kernel_size=(3,3), strides=(2, 2), activation='relu'),
                # tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Flatten(),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=(3,3), strides=2, padding='same',
                activation='relu'),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=(3,3), strides=2, padding='same',
                activation='relu'),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2DTranspose(
                filters=16, kernel_size=(3,3), strides=1, padding='same',
                activation='relu'),
            # tf.keras.layers.BatchNormalization(),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=3, kernel_size=(3,3), strides=1, padding='same'),
        ]
    )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    optimizer = tf.keras.optimizers.Adam(1e-4)
    def log_normal_pdf(sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),axis=raxis)

    def compute_loss(model, x):
        mean, logvar = model.encode(x)
        z = model.reparameterize(mean, logvar)
        x_logit = model.decode(z)
        #tf.cast(x, tf.float32).
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = CVAE.log_normal_pdf(z, 0., 0.)
        logqz_x = CVAE.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)


    @tf.function
    def train_step(model, x, optimizer):
        """Executes one training step and returns the loss.

        This function computes the loss and gradients, and `use`s the latter to
        update the model's parameters.
        """
        with tf.GradientTape() as tape:
            loss = CVAE.compute_loss(model, x)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    def generate_and_save_images(model, epoch, test_sample, log_dir='', title=''):
        mean, logvar = model.encode(test_sample)
        z = model.reparameterize(mean, logvar)
        predictions = model.sample(z)
        fig = plt.figure(figsize=(4, 4))
        print("predictions shape: {}".format(predictions.shape))
        for i in range(predictions.shape[0]):
            #save using pil
            x_recover = np.rint(predictions[i, :, :, :]*255).astype(np.uint8)
            pil_image1=Image.fromarray(x_recover, mode='RGB')
            #
            plt.subplot(4, 4, i + 1)
            # plt.imshow(predictions[i, :, :, :])
            plt.imshow(pil_image1)
            plt.axis('off')
        # tight_layout minimizes the overlap between 2 sub-plots
        plt.savefig(log_dir + '/{}_epoch_{:04d}.png'.format(title, epoch))
        plt.close()

def vae_training(iteration, client, model_name='vae_model'):
    #training vae
    for iter in range(iteration):
        for train_x in client['vae_train_dataset']:
            CVAE.train_step(client[model_name], train_x, CVAE.optimizer)
        loss = tf.keras.metrics.Mean()
        for i,test_x in enumerate(client['vae_test_dataset']):
            if i <10:
                loss(CVAE.compute_loss(client[model_name], test_x))
            else:
                break
        elbo = -loss.result()
        # logging
        print('Epoch: {}, Test set ELBO: {}'.format(iter, elbo))
    return client



if __name__=='__main__':
    ## testing VAE model

    X_train, y_train, X_test, y_test = prepare_data(dataset='SVHN')
    epochs = 200
    batch_size = 32
    # set the dimensionality of the latent space to a plane for visualization later
    latent_dim = 50
    num_examples_to_generate = 16

    # keeping the random vector constant for generation (prediction) so
    # it will be easier to see the improvement.
    random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, latent_dim])
    model = CVAE(latent_dim)

    train_dataset = (tf.data.Dataset.from_tensor_slices(X_train)
                 .shuffle(X_train.shape[0]).batch(batch_size))
    test_dataset = (tf.data.Dataset.from_tensor_slices(X_test)
                .shuffle(X_test.shape[0]).batch(batch_size))

    # Pick a sample of the test set for generating output images
    assert batch_size >= num_examples_to_generate
    for test_batch in test_dataset.take(1):
        test_sample = test_batch[0:num_examples_to_generate, :, :, :]
    CVAE.generate_and_save_images(model, 0, test_sample)
    def display_image(epoch_no):
        return Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))


    ## training the model
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        for train_x in train_dataset:
            CVAE.train_step(model, train_x, CVAE.optimizer)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(CVAE.compute_loss(model, test_x))
        elbo = -loss.result()

        # logging
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'\
            .format(epoch, elbo, end_time - start_time))
        if epoch%10==0: CVAE.generate_and_save_images(model, epoch, test_sample)

   