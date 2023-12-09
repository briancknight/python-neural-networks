import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist

from net.layers import Dense, Reshape, Conv, ConvTranspose, ConvResBlock
from net.activations import Tanh, ReLU, Softmax
from net.losses import MSE, BinaryCrossEntropy
from net.optimizers import SGD, RMSprop
from net.initializers import Xavier
from net.utils import create_model, train, test, forward

# some helpers for loading MNIST
def to_categorical(labels):
    cats = np.unique(labels)
    categorical_labels = np.zeros((len(labels), len(cats)))
    for j in range(len(labels)):
        idx = 0
        while sum(categorical_labels[j]) == 0:
            if labels[j] == cats[idx]:
                categorical_labels[j][idx] = 1.0
            idx += 1
    return categorical_labels

def load_data(n):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_train /= 255
    x_train = x_train.reshape(len(x_train), 28, 28, 1)
    y_train = to_categorical(y_train)

    x_test = x_test.astype('float32')
    x_test = x_test.reshape(len(x_test), 28, 28, 1)
    x_test /= 255
    y_test = to_categorical(y_test)

    return x_train[:n], y_train[:n], x_test, y_test

cae = create_model([
    Conv(1, (3,3), input_shape = (28, 28, 1), output_shape = (26, 26, 1)),
    Tanh(),
    Conv(1, (3,3), input_shape = (26, 26, 1), output_shape = (24, 24, 1)),
    Tanh(),
    Conv(1, (5,5), input_shape = (24, 24, 1), output_shape = (20, 20, 1)),
    Tanh(),
    ConvTranspose(1, (5,5), input_shape = (20, 20, 1), output_shape = (24, 24, 1)),
    Tanh(),
    ConvTranspose(1, (3,3), input_shape = (24, 24, 1), output_shape = (26, 26, 1)),
    Tanh(),
    ConvTranspose(1, (3,3), input_shape = (26, 26, 1), output_shape = (28, 28, 1)),
], Xavier(), RMSprop, {'learning_rate': 0.1})
loss = MSE()

x_train, y_train, x_test, y_test = load_data(1000)
train(cae, loss, x_train, x_train, epochs=50)
print('error on test set:', test(cae, loss, x_test, x_test))


encoder = cae[:6]
decoder = cae[6:]
n = 20
f, ax = plt.subplots(5, 3)
for i in range(5):
    code, transport1 = forward(encoder, x_train[i])
    reconstructed = forward(decoder, code)[0][:,:,0]
    ax[i][0].imshow(x_train[i], cmap='gray')
    ax[i][1].imshow(code[:,:,0], cmap='gray')
    ax[i][2].imshow(reconstructed, cmap='gray')
plt.show()

# encode the originial training and testing data via the encoder we just trained:

encoded_train = []
for x in x_train:
    encoded_train.append(forward(encoder, x)[0])
    
encoded_test = []
for x in x_test:
    encoded_test.append(forward(encoder, x)[0])
    
""" 
now train a classifier which uses a LAP regularizer to encourage optimal transport 
through the convolutional residual blocks
"""

lap_classifier = create_model([
    ConvResBlock((3,3), input_shape=(n,n,1), output_shape=(n,n,1)),
    ConvResBlock((3,3), input_shape=(n,n,1), output_shape=(n,n,1)),
    Reshape((1,n**2)),
    Dense(128),
    Tanh(),
    Dense(10),
    Softmax()
], Xavier(), SGD, {'learning_rate': 0.1})
mse = MSE()

x_train, y_train, x_test, y_test = load_data(1000)
lmbda = 15
print('error here includes some measure of average transport costs')
train(lap_classifier, mse, encoded_train, y_train, epochs=15, lmbda=lmbda)
# print('error on test set:', test(lap_classifier, mse, encoded_test, y_test))



f, ax = plt.subplots(5, 4)
for i in range(5):
    code = forward(encoder, x_test[i])[0]
    # reconstructed = forward(decoder, code)[0]
    block1 = forward(lap_classifier[:1], code)[0]
    block2 = forward(lap_classifier[:2], code)[0]
    ax[i][0].imshow(x_test[i], cmap='gray')
    ax[i][1].imshow(code[:,:,0], cmap='gray')
    ax[i][2].imshow(block1[:,:,0], cmap='gray')
    ax[i][3].imshow(block2[:,:,0], cmap='gray')

plt.show()


f, ax = plt.subplots(5, 5)
for i in range(5):
    code = forward(encoder, x_test[i])[0]
    block1 = forward(lap_classifier[:1], code)[0]
    block2 = forward(lap_classifier[:2], code)[0]
    reconstructed1 = forward(decoder, code)[0]
    reconstructed2 = forward(decoder, block1)[0]
    reconstructed3 = forward(decoder, block2)[0]
    # reconstructed6 = forward(decoder, forward(model2[:5], code1)[0].reshape(1,16))[0]
    ax[i][0].imshow(x_test[i], cmap='gray')
    ax[i][1].imshow(code[:,:,0], cmap='gray')
    ax[i][2].imshow(reconstructed1[:,:,0], cmap='gray')
    ax[i][3].imshow(reconstructed2[:,:,0], cmap='gray')
    ax[i][4].imshow(reconstructed3[:,:,0], cmap='gray')

plt.show()


error = 0
transports = 0
for x, y in zip(encoded_test, y_test):
    (output, transport) = forward(lap_classifier, x)
    transports+= transport
    error += loss.call(y, output[0]) #+ lmbda * transport
error /= len(x_test)
transports /= len(x_test)

print('classification error = %f, average transport cost across residual blocks = %f ' % (error, transports/lmbda))