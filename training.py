from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import numpy as np
# import sklearn.datasets
import tensorflow as tf
import os
import sys
import utils
from librispeech_mixer import LibriSpeechMixer

import keras
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose
from tensorflow.contrib.layers import flatten # We use this flatten, as it works better than
                                              # the Keras 'Flatten' for some reason

tf.reset_default_graph()

height, width, nchannels = 129, 200, 1
padding = 'same'

filters_1 = 16
kernel_size_1 = (129,20)
pool_size_1 = (2,2)

x_pl = tf.placeholder(tf.float32, [None, height, width, nchannels], name='xPlaceholder')
y_pl = tf.placeholder(tf.float64, [None, height, width, 2], name='yPlaceholder')
y_pl = tf.cast(y_pl, tf.float32)

print('Trace of the tensors shape as it is propagated through the network.')
print('Layer name \t Output size')
print('----------------------------')

with tf.variable_scope('convLayer1'):
    conv1 = Conv2D(filters_1, kernel_size_1, strides=(1,1), padding=padding, activation='relu')
    print('x_pl \t\t', x_pl.get_shape())
    x = conv1(x_pl)
    print('conv1 \t\t', x.get_shape())

    """pool1 = MaxPooling2D(pool_size=pool_size_1, strides=None, padding=padding)
    x = pool1(x)
    print('pool1 \t\t', x.get_shape())
    x = flatten(x)
    print('Flatten \t', x.get_shape())"""



with tf.variable_scope('output_layer'):
    deconv = Conv2DTranspose(2, kernel_size_1, strides=(1,1), padding=padding, activation='relu')

    y = deconv(x)
    print('deconv\t', y.get_shape())

print('Model consits of ', utils.num_params(), 'trainable parameters.')

## Launch TensorBoard, and visualize the TF graph
gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.45)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
    tmp_def = utils.rename_nodes(sess.graph_def, lambda s:"/".join(s.split('_',1)))
    utils.show_graph(tmp_def)


with tf.variable_scope('loss'):
    # computing cross entropy per sample
    cross_entropy = -tf.reduce_sum(y_pl * tf.log(y+1e-8), reduction_indices=[1])

    # averaging over samples
    cross_entropy = tf.reduce_mean(cross_entropy)


with tf.variable_scope('training'):
    # defining our optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    # applying the gradients
    train_op = optimizer.minimize(cross_entropy)


with tf.variable_scope('performance'):
    # making a one-hot encoded vector of correct (1) and incorrect (0) predictions
    correct_prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pl, axis=1))

    # averaging the one-hot encoded vector
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


#Create the LibriSpeech mixer
mixer = LibriSpeechMixer()

#Test the forward pass
x_batch, y_batch = mixer.get_batch()

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
# with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_pred = sess.run(fetches=y, feed_dict={x_pl: x_batch})

assert y_pred.shape == y_batch.shape, "ERROR the output shape is not as expected!"         + " Output shape should be " + str(y.shape) + ' but was ' + str(y_pred.shape)

print('Forward pass successful!')


# ## Training

# In[ ]:

#Training Loop
batch_size = 100
max_epochs = 10


valid_loss, valid_accuracy = [], []
train_loss, train_accuracy = [], []
test_loss, test_accuracy = [], []


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('Begin training loop')

    try:
        while mnist_data.train.epochs_completed < max_epochs:
            _train_loss, _train_accuracy = [], []

            ## Run train op
            x_batch, y_batch = mnist_data.train.next_batch(batch_size)
            fetches_train = [train_op, cross_entropy, accuracy]
            feed_dict_train = {x_pl: x_batch, y_pl: y_batch}
            _, _loss, _acc = sess.run(fetches_train, feed_dict_train)

            _train_loss.append(_loss)
            _train_accuracy.append(_acc)


            ## Compute validation loss and accuracy
            if mnist_data.train.epochs_completed % 1 == 0                     and mnist_data.train._index_in_epoch <= batch_size:
                train_loss.append(np.mean(_train_loss))
                train_accuracy.append(np.mean(_train_accuracy))

                fetches_valid = [cross_entropy, accuracy]

                feed_dict_valid = {x_pl: mnist_data.validation.images, y_pl: mnist_data.validation.labels}
                _loss, _acc = sess.run(fetches_valid, feed_dict_valid)

                valid_loss.append(_loss)
                valid_accuracy.append(_acc)
                print("Epoch {} : Train Loss {:6.3f}, Train acc {:6.3f},  Valid loss {:6.3f},  Valid acc {:6.3f}".format(
                    mnist_data.train.epochs_completed, train_loss[-1], train_accuracy[-1], valid_loss[-1], valid_accuracy[-1]))


        test_epoch = mnist_data.test.epochs_completed
        while mnist_data.test.epochs_completed == test_epoch:
            x_batch, y_batch = mnist_data.test.next_batch(batch_size)
            feed_dict_test = {x_pl: x_batch, y_pl: y_batch}
            _loss, _acc = sess.run(fetches_valid, feed_dict_test)
            test_loss.append(_loss)
            test_accuracy.append(_acc)
        print('Test Loss {:6.3f}, Test acc {:6.3f}'.format(
                    np.mean(test_loss), np.mean(test_accuracy)))


    except KeyboardInterrupt:
        pass


# In[ ]:

epoch = np.arange(len(train_loss))
plt.figure()
plt.plot(epoch, train_accuracy,'r', epoch, valid_accuracy,'b')
plt.legend(['Train Acc','Val Acc'], loc=4)
plt.xlabel('Epochs'), plt.ylabel('Acc'), plt.ylim([0.75,1.03])


# # Assignments

# #### <span style="color:red"> EXE 1.1 </span> Manual calculations
#
# ![](images/conv_exe.png)
#
#
#
# 1. Manually convolve the input, and compute the convolved features. No padding and no strieds.
# 1. Perform `2x2` max pooling on the convolved features. Stride of 2.
#
# ___
#
# <span style="color:blue"> Answer: </span>
#
#
#
#

#
# #### <span style="color:red"> EXE 1.2 </span> Reducing the resolution
# One of the important features of convolutional networks are their ability to reduce the spatial resolution, while retaining the important features.
# Effectively this gives a local translational invariance and reduces the computation.
# This is most often done with **maxpooling** or by using strides.
#
# 1. Using only convolutional layers and pooling operations reduce the feature map size to `1x1xF`.
#     * The number of feature maps, `F`, is up to you.
#
# ___
#
# <span style="color:blue"> Write down what you did: </span>
#
# ```
# Paste your code here
# ```
#
#
# ```
# Paste the trace of the tensors shape as it is propagated through the network here
# ```
#

# #### <span style="color:red"> EXE 1.3 </span> Play around with the network.
# The MNIST dataset is so easy to solve with convolutional networks that it isn't interesting to spend to much time on maximizing performance.
# A more interesting question is *how few parameters can you solve it with?*
#
# 1. Try and minimize the number of parameters, while keeping validation accuracy about 95%. Try changing the
#
#     * Number of layers
#     * Number of filters
#     * Kernel size
#     * Pooling size
# 1. Once happy take note of the performance, number of parameters (printed automatically), and describe the network below.
# ___
#
#
# <span style="color:blue"> Answer: </span>
#

# #### <span style="color:red"> EXE 1.4 </span> Comparing dense and convolutional networks
#
# 1. Now create a densely connected network (the ones from lab 1), and see how good performance you can get with a similar number of parameters.
# ___
#
# <span style="color:blue"> Describe your findings: </span>
#
