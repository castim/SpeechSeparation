from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import utils
from scipy.signal import spectrogram, istft
from librispeech_mixer import LibriSpeechMixer
from keras.layers import Input, Dense, Conv1D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Reshape, Flatten, Dropout, BatchNormalization
from tensorflow.contrib.layers import flatten
import IPython
from os import listdir
from keras import backend as K


def mask_to_outputs(mixed_real, mixed_imag, scaled_mask_real, scaled_mask_imag, C, K):
    unscaled_mask_real = -tf.log((K - scaled_mask_real)/(K + scaled_mask_real+1e-10)+1e-30)/C
    unscaled_mask_imag = -tf.log((K - scaled_mask_imag)/(K + scaled_mask_imag+1e-10)+1e-30)/C

    sep1 = tf.multiply(tf.complex(unscaled_mask_real, unscaled_mask_imag), tf.complex(mixed_real, mixed_imag))
    sep2 = tf.complex(mixed_real, mixed_imag) - sep1

    return sep1, sep2

tf.reset_default_graph()
K.set_learning_phase(1) #set learning phase

#Create the LibriSpeech mixer
mixer = LibriSpeechMixer(dataset_built=True)

#parse function to get data from the dataset correctly
def _parse_function(example_proto):
    keys_to_features = {'mixed_real':tf.FixedLenFeature((mixer.spec_length, mixer.nb_freq), tf.float32),
                        'mixed_imag':tf.FixedLenFeature((mixer.spec_length, mixer.nb_freq), tf.float32),
                        'mask_real': tf.FixedLenFeature((mixer.spec_length, mixer.nb_freq), tf.float32),
                        'mask_imag': tf.FixedLenFeature((mixer.spec_length, mixer.nb_freq), tf.float32),
                        }
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)
    return tf.concat([parsed_features['mixed_real'], parsed_features['mixed_imag']], axis=1),\
            tf.concat([parsed_features['mask_real'], parsed_features['mask_imag']], axis=1)

#Create the dataset object
batch_size = 64

#Placeholder to be able to specify either the training or validation set
filenames = tf.placeholder(tf.string, shape=[None])
dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(_parse_function)
dataset = dataset.shuffle(buffer_size=2500)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()
iterator = dataset.make_initializable_iterator()
x_pl, y_pl = iterator.get_next()

training_filenames = ["/mnt/train/" + filename for filename in listdir("/mnt/train/")]
validation_filenames = ["/mnt/dev/" + filename for filename in listdir("/mnt/dev/")]


height, width, nchannels = mixer.nb_freq, mixer.spec_length, 1
padding = 'same'

filters = mixer.nb_freq*2
kernel_size = 3

print('Trace of the tensors shape as it is propagated through the network.')
print('Layer name \t Output size')
print('----------------------------')

with tf.variable_scope('convLayer1'):

    conv1 = Conv1D(round(filters), kernel_size, padding=padding, activation='relu')
    print('x_pl \t\t', x_pl.get_shape())
    x = conv1(x_pl)
    print('conv1 \t\t', x.get_shape())

    conv2 = Conv1D(round(filters), kernel_size, padding=padding, activation='relu')
    x = conv2(x)
    print('conv2 \t\t', x.get_shape())

    conv3 = Conv1D(round(filters), kernel_size, padding=padding, activation='relu')
    x = conv3(x)
    print('conv3 \t\t', x.get_shape())

    conv4 = Conv1D(round(filters), kernel_size, padding=padding, activation='relu')
    x = conv4(x)
    print('conv4 \t\t', x.get_shape())
    enc_cell = tf.nn.rnn_cell.GRUCell(mixer.nb_freq*2, activation = tf.nn.relu)
    x, enc_state = tf.nn.dynamic_rnn(cell=enc_cell, inputs=x,
                                     dtype=tf.float32)

    convend = Conv1D(round(filters), 1, padding=padding, activation='tanh')
    y = convend(x)
    print('convend \t\t', x.get_shape())

    y = y * mixer.K
print('Model consits of ', utils.num_params(), 'trainable parameters.')
# restricting memory usage, TensorFlow is greedy and will use all memory otherwise
gpu_opts = tf.GPUOptions(per_process_gpu_memory_fraction=0.99)

with tf.variable_scope('loss'):
    # The loss takes the amplitude of the output into account, in order to avoid taking care of noise
    y_target1, y_target2 = mask_to_outputs(x_pl[:, :, :mixer.nb_freq], x_pl[:, :, mixer.nb_freq:],\
                                            y_pl[:, :, :mixer.nb_freq], y_pl[:, :, mixer.nb_freq:], mixer.C, mixer.K)

    y_pred1, y_pred2 = mask_to_outputs(x_pl[:, :, :mixer.nb_freq], x_pl[:, :, mixer.nb_freq:],\
                                            y[:, :, :mixer.nb_freq], y[:, :, mixer.nb_freq:], mixer.C, mixer.K)

    mean_square_error = tf.reduce_mean((tf.real(y_target1) - tf.real(y_pred1))**2 + \
                                        (tf.real(y_target2) - tf.real(y_pred2))**2 + \
                                        (tf.imag(y_target1) - tf.imag(y_pred1))**2 + \
                                        (tf.imag(y_target2) - tf.imag(y_pred2))**2)

with tf.variable_scope('training'):
    # defining our optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

    # applying the gradients
    train_op = optimizer.minimize(mean_square_error)

#Test the forward pass
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
    sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
    sess.run(tf.global_variables_initializer())
    y_pred = sess.run(fetches=y)

assert y_pred.shape[1:] == y_pl.shape[1:], "ERROR the output shape is not as expected!"         + " Output shape should be " + str(y_pl.shape) + ' but was ' + str(y_pred.shape)

print('Forward pass successful!')

# ## Training

#Training Loop

max_epochs = 100


valid_loss = []
train_loss = []
test_loss = []


def trainingLoop():
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_opts)) as sess:
        saver = tf.train.Saver()
        sess.run(iterator.initializer, feed_dict={filenames: training_filenames})
        #saver.restore(sess, "complex3filtersdropout.ckpt")
        #print("Model restored.")
        sess.run(tf.global_variables_initializer())
        print('Begin training loop')

        nb_batches_processed = 0
        nb_epochs = 0
        _train_loss = []
        try:

            while nb_epochs < max_epochs:

                ## Run train op
                fetches_train = [train_op, mean_square_error]

                _, _loss = sess.run(fetches_train)
                _train_loss.append(_loss)

                nb_batches_processed += 1

                ## Compute validation loss once per epoch
                if round(nb_batches_processed/mixer.nb_seg_train*batch_size-0.5) > nb_epochs:
                    nb_epochs += 1

                    sess.run(iterator.initializer, feed_dict={filenames: validation_filenames})
                    _valid_loss = []
                    train_loss.append(np.mean(_train_loss))
                    _train_loss = []

                    fetches_valid = [mean_square_error]

                    nb_test_batches_processed = 0
                    #Proceed to a whole testing epoch
                    while round(nb_test_batches_processed/mixer.nb_seg_test*batch_size-0.5) < 1:

                        _loss = sess.run(fetches_valid)

                        _valid_loss.append(_loss)
                        nb_test_batches_processed += 1

                    valid_loss.append(np.mean(_valid_loss))


                    print("Epoch {} : Train Loss {:6.6f}, Valid loss {:6.6f}".format(
                        nb_epochs, train_loss[-1], valid_loss[-1]))
                    sess.run(iterator.initializer, feed_dict={filenames: training_filenames})

        except KeyboardInterrupt:
            pass

        save_path = saver.save(sess, "./complex3filters5convsamefilters.ckpt")
        print("Model saved");


trainingLoop();
