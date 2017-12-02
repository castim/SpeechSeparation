from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import utils
from scipy.signal import spectrogram, istft
from test_mixer import TestMixer
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Reshape, Flatten, Dropout, BatchNormalization
from tensorflow.contrib.layers import flatten
import IPython
from os import listdir
from keras import backend as K

#Create the LibriSpeech mixer
mixer = TestMixer(dataset_built=False)