# Autoencoder encoder with dropout for MNIST handwritten digit dataset

# To run this script on enmcomp3, 4 and 11 with GPU, type the following
# bash
# export PATH=/usr/local/anaconda3/bin:/usr/local/cuda-8.0/bin:$PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/cuda-7.5/lib64
# source activate tf-py3.6
# python3 mnist_autoencoder.py
# source deactivate tf-py3.6

from keras.layers import Input, Dense, Dropout
from keras.models import Model
from keras.datasets import mnist
from keras.constraints import maxnorm
from keras import optimizers
import numpy as np
from sklearn.manifold import TSNE
from myPlot import scatter2D
import matplotlib.pyplot as plt
import tensorflow as tf

# Dimensions of input and encoded representations
input_dim = 784
encoding_dim = 64

# Use 1/3 of the GPU memory so that the GPU can be shared by multiple users
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# This is the input placeholder
input_img = Input(shape=(input_dim,))

# Define the structure of the autoencoder. Add a dropout layer between the input
# and the first hidden layer. The dropout rate is set to 20%, meaning one in 5 inputs
# will be randomly excluded from each update cycle. a constraint is imposed on the weights for
# each hidden layer, ensuring that the maximum norm of the weights does not exceed a value of 3.
# "encoded" is the encoded representation of the input
encoded = Dropout(0.2)(input_img)
encoded = Dense(256, activation='relu', kernel_constraint=maxnorm(3))(encoded)
encoded = Dropout(0.2)(encoded)
encoded = Dense(128, activation='relu', kernel_constraint=maxnorm(3))(encoded)
encoded = Dropout(0.2)(encoded)
encoded = Dense(encoding_dim, activation='linear', kernel_constraint=maxnorm(3))(encoded)

# "decoded" is the lossy reconstruction of the input.
# Need to use 'sigmoid' for the last layer because 'binary_crossentropy' is
# used as the loss function
decoded = Dropout(0.2)(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dropout(0.2)(decoded)
decoded = Dense(256, activation='relu')(encoded)
decoded = Dropout(0.2)(decoded)
decoded = Dense(input_dim, activation='sigmoid')(decoded)

# this model maps an input to its reconstruction
autoencoder = Model(inputs=input_img, outputs=decoded)

# Compile the model. Better reconstructions are obtained if 'binary_crossentropy' is used
# Because dropout is used, larger learning rate and momentum can be used to speed up training
#optimizers.SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)
opt = optimizers.Adadelta(lr=10, rho=0.95, epsilon=1e-08, decay=0.0)
autoencoder.compile(optimizer=opt, loss='binary_crossentropy')

# This model maps an input to its encoded representation
encoder = Model(inputs=input_img, outputs=encoded)

# Load and normalize MNIST digits
(x_train, trn_lbs), (x_test, tst_lbs) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

# Train the autoencoder
autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True,
                validation_data=(x_test, x_test))

# Encode and decode the test digits
encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)

# Use t-SNE to project and plot the first 100 encoded_imgs of each digit
print('Peforming t-SNE...')
n_imgs = 100
X = np.empty([0, encoded_imgs.shape[1]], dtype='float32')
y = np.array([], dtype='float32')
for i in range(10):
    select_imgs = encoded_imgs[tst_lbs == i]
    select_imgs = select_imgs[0:n_imgs, :]
    X = np.vstack((X, select_imgs))
    y = np.concatenate((y, i*np.ones(n_imgs)))

Xproj = TSNE(random_state=20150101).fit_transform(X)
print('Plot encoded digits on 2-D space using t-SNE')
scatter2D(Xproj, y)

# Plot some orignial digits and their corresponding transcoded versions
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()



