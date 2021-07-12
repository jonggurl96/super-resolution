import numpy as np
import tensorflow as tf
import tensorflow.keras
import cv2, os

class DataGeneratorYUV(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.dim[0] * 4, self.dim[1] * 4, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            a = np.load(ID)
            a = tf.image.rgb_to_yuv(a)
            last_dimension_axis = len(a.shape) - 1
            Y, U, V = tf.split(a, 3, axis=last_dimension_axis)
            X[i] = tf.image.resize(Y, [44, 44], method="area")

            splited = ID.split('\\')
            splited[-2] = 'y' + splited[-2][1:] # x_train -> y_train
            y_path = os.path.join(os.sep, *splited)

            # Store class
            b = np.load(y_path)
            b = tf.image.rgb_to_yuv(b)
            last_dimension_axis = len(b.shape) - 1
            Y, U, V = tf.split(b, 3, axis=last_dimension_axis)
            y[i] = Y

        return X, y