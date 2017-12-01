import os, glob
import numpy as np
from scipy.signal import spectrogram, stft, istft
import random
from pydub import AudioSegment
from multiprocessing import Pool
import tensorflow as tf

class TestMixer:

    #The length of the spectrogram we take
    spec_length = 128
    nb_freq = 128

    def __init__(self, nbSamples = float("inf"), nbSpeakers = float("inf"), dataset_built=True):
        self.male_audios = []
        self.female_audios = []
        self.indices = []

        audio_dir = "Data/LibriSpeech/train-clean-100/"

        female_file = "magnolia/data/librispeech/authors/train-clean-100-F.txt"
        male_file = "magnolia/data/librispeech/authors/train-clean-100-M.txt"

        self.data_path_train = "/mnt/train/spec"

        self.data_path_test = "/mnt/dev/spec"

        #Collect males dirs:
        male_speaker_dirs = [];
        with open(male_file, "r") as f:

            i = 0
            for folder in f:
                folder = folder[:-1]
                male_speaker_dirs[0:0] = glob.glob(os.path.join(audio_dir + folder, '*'))
                i += 1
                if i >= nbSpeakers:
                    break;

        #Collect females files:
        female_speaker_dirs = [];
        with open(female_file, "r") as f:

            i= 0
            for folder in f:
                folder = folder[:-1]
                female_speaker_dirs[0:0] = glob.glob(os.path.join(audio_dir + folder, '*'))
                i += 1
                if i >= nbSpeakers:
                    break;

        for male_dir in male_speaker_dirs:

            self.male_audios[0:0] = glob.glob(os.path.join(male_dir, '*.flac'))

        self.male_audios = np.random.permutation(self.male_audios)

        for female_dir in female_speaker_dirs:
            self.female_audios[0:0] = glob.glob(os.path.join(female_dir, '*.flac'))

        self.female_audios = np.random.permutation(self.female_audios)

        maxInd = min(nbSamples, len(self.male_audios), len(self.female_audios))
        self.indices = range(0,maxInd)

        #Define the training/test sets split
        self.sep = round(0.8*maxInd)
        self.indices_train = self.indices[:self.sep]
        self.indices_test = self.indices[self.sep:]

        if not dataset_built:
            self.build_dataset_tfrecord()

    def mix_and_save_record(self, indices, filename):

        writer = tf.python_io.TFRecordWriter(filename)
        for i, j in indices:
            sound1 = AudioSegment.from_file(self.male_audios[i], format='flac')
            target1 = self.normalise_divmax(np.array(sound1.get_array_of_samples()))

            sound2 = AudioSegment.from_file(self.female_audios[j],format='flac')
            target2 = self.normalise_divmax(np.array(sound2.get_array_of_samples()))

            length = min(len(target1), len(target2))

            freqs_target1, bins_target1, Pxx_target1 = stft(target1[:length])
            freqs_target2, bins_target2, Pxx_target2 = stft(target2[:length])
            mask_target = np.abs(Pxx_target1) / (np.abs(Pxx_target2) + np.abs(Pxx_target1) + 1e-100)

            Fxx_mixed = Pxx_target1 + Pxx_target2

            #slice the sample
            for k in range(0,Fxx_mixed.shape[1]//self.spec_length):
                in_spec = np.moveaxis(np.array([Fxx_mixed])[:, :self.nb_freq, k*self.spec_length:(k+1)*self.spec_length], 0, -1)
                mask = np.moveaxis(np.array([mask_target])[:, :self.nb_freq, k*self.spec_length:(k+1)*self.spec_length], 0, -1)

                example = tf.train.Example(features=tf.train.Features(feature={
                    'mixed_abs': tf.train.Feature(float_list=tf.train.FloatList(value=np.abs(in_spec).flatten())),
                    'mixed_phase': tf.train.Feature(float_list=tf.train.FloatList(value=np.angle(in_spec).flatten())),
                    'mask': tf.train.Feature(float_list=tf.train.FloatList(value=mask.flatten()))}))

                writer.write(example.SerializeToString())
        writer.close()

    def build_dataset_tfrecord(self):

        indices_iterator = list(self.indices_train)
        random.shuffle(indices_iterator)

        indices = list(enumerate(indices_iterator))
        p = Pool(9)
        n_files_record = 50
        #Create records using 50 files in each
        #enumerate to take different females for the males
        p.starmap(self.mix_and_save_record, [(indices[n_files_record*k:n_files_record*k+n_files_record],\
            self.data_path_train + str(k)+ ".tfrecords")\
            for k in range(0,round(len(indices_iterator)/n_files_record + 0.5))])


        indices_iterator = list(self.indices_test)
        random.shuffle(indices_iterator)

        indices = list(enumerate(indices_iterator))
        #enumerate to take different females for the males
        p.starmap(self.mix_and_save_record, [(indices[n_files_record*k:n_files_record*k+n_files_record],\
            self.data_path_test + str(k)+ ".tfrecords")\
            for k in range(0,round(len(indices_iterator)/n_files_record + 0.5))])

    def normalise_divmax(self, samples):

        normalised = samples / max(samples)

        return normalised
