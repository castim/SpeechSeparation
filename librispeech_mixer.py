import os, glob
import numpy as np
from scipy.signal import spectrogram
import random
from pydub import AudioSegment

class LibriSpeechMixer:

    #If you want to avoid scratching the ssd
    #sudo mount -t tmpfs -o size=500m tmpfs tmpMixed
    output_dir = "tmpMixed/"

    #The length of the spectrogram we take
    spec_length = 64
    nb_freq = 128

    # Difference in the speech signal levels in dB.
    # Assumes that both audio files have been correctly normalized and have the same speech signal level initially.

    def __init__(self, train=True, nbSamples = float("inf")):
        self.male_audios = []
        self.female_audios = []
        self.indices = []
        self.indices_it = None
        self.epochs_completed = 0
        self.index_in_epoch = 0

        if train:
            audio_dir = "Data/LibriSpeech/train-clean-100/"

            female_file = "magnolia/data/librispeech/authors/train-clean-100-F.txt"
            male_file = "magnolia/data/librispeech/authors/train-clean-100-M.txt"

            self.in_data_path = "Data/train/in/spec"
            self.out_data_path = "Data/train/out/spec"
        else:
            audio_dir = "Data/LibriSpeech/dev-clean/"

            female_file = "magnolia/data/librispeech/authors/dev-clean-F.txt"
            male_file = "magnolia/data/librispeech/authors/dev-clean-M.txt"

            self.in_data_path = "Data/dev/in/spec"
            self.out_data_path = "Data/dev/out/spec"

        #Collect males dirs:
        male_speaker_dirs = [];
        with open(male_file, "r") as f:

            for folder in f:
                folder = folder[:-1]
                male_speaker_dirs[0:0] = glob.glob(os.path.join(audio_dir + folder, '*'))

        #Collect females files:
        female_speaker_dirs = [];
        with open(female_file, "r") as f:

            for folder in f:
                folder = folder[:-1]
                female_speaker_dirs[0:0] = glob.glob(os.path.join(audio_dir + folder, '*'))

        for male_dir in male_speaker_dirs:

            self.male_audios[0:0] = glob.glob(os.path.join(male_dir, '*.flac'))

        self.male_audios = np.random.permutation(self.male_audios)

        for female_dir in female_speaker_dirs:
            self.female_audios[0:0] = glob.glob(os.path.join(female_dir, '*.flac'))

        self.female_audios = np.random.permutation(self.female_audios)

        self.indices = range(0,min(nbSamples, len(self.male_audios), len(self.female_audios)))

        #The list function performs a shallow copy
        self.indices_it = list(self.indices)

        #works in place
        random.shuffle(self.indices_it)

        self.indices_it = iter(self.indices_it)

    def next(self):

        try:
            i = next(self.indices_it)
            self.index_in_epoch += 1

        except StopIteration:
            #The list function performs a shallow copy
            self.indices_it = list(self.indices)

            #works in place
            random.shuffle(self.indices_it)
            self.indices_it = iter(self.indices_it)

            self.epochs_completed += 1
            self.index_in_epoch = 0
            i = next(self.indices_it)

        sound1 = AudioSegment.from_file(self.male_audios[i], format='flac')
        target1 = self.normalise_divmax(np.array(sound1.get_array_of_samples()))

        sound2 = AudioSegment.from_file(self.female_audios[i],format='flac')
        target2 = self.normalise_divmax(np.array(sound2.get_array_of_samples()))

        output = sound1.overlay(sound2, position=0)
        mixed = self.normalise_divmax(np.array(output.get_array_of_samples()))

        length = min(len(target1), len(target2))

        freqs_target1, bins_target1, Pxx_target1 = spectrogram(target1[:length])
        freqs_target2, bins_target2, Pxx_target2 = spectrogram(target2[:length])
        mask_target = Pxx_target1 / (Pxx_target2 + Pxx_target1 + 1e-100)

        freqs_mixed, bins_mixed, Pxx_mixed = spectrogram(mixed[:length])

        return np.moveaxis(np.array([Pxx_mixed])[:, :, :self.spec_length], 0, -1), \
               np.moveaxis(np.array([mask_target])[:, :, :self.spec_length], 0, -1)

    def build_dataset(self):

        for i in self.indices:

            sound1 = AudioSegment.from_file(self.male_audios[i], format='flac')
            target1 = self.normalise_divmax(np.array(sound1.get_array_of_samples()))

            sound2 = AudioSegment.from_file(self.female_audios[i],format='flac')
            target2 = self.normalise_divmax(np.array(sound2.get_array_of_samples()))

            output = sound1.overlay(sound2, position=0)
            mixed = self.normalise_divmax(np.array(output.get_array_of_samples()))

            length = min(len(target1), len(target2))

            freqs_target1, bins_target1, Pxx_target1 = spectrogram(target1[:length])
            freqs_target2, bins_target2, Pxx_target2 = spectrogram(target2[:length])
            mask_target = Pxx_target1 / (Pxx_target2 + Pxx_target1 + 1e-100)

            freqs_mixed, bins_mixed, Pxx_mixed = spectrogram(mixed[:length])

            np.save(self.in_data_path + str(i), np.moveaxis(np.array([Pxx_mixed])[:, :, :self.spec_length], 0, -1))
            np.save(self.out_data_path + str(i), np.moveaxis(np.array([mask_target])[:, :, :self.spec_length], 0, -1))

    def load_dataset(self):
        self.in_data = np.empty((len(self.indices), self.nb_freq, self.spec_length, 1))
        self.out_data = np.empty((len(self.indices), self.nb_freq, self.spec_length, 1))

        for i in self.indices:
            self.in_data[i, :, :, :] = np.load(self.in_data_path + str(i) + ".npy")[:self.nb_freq, :, :]
            self.out_data[i, :, :, :] = np.load(self.out_data_path + str(i) + ".npy")[:self.nb_freq, :, :]


    def next_load_file(self):
        try:
            i = next(self.indices_it)
            self.index_in_epoch += 1

        except StopIteration:
            #The list function performs a shallow copy
            self.indices_it = list(self.indices)

            #works in place
            random.shuffle(self.indices_it)
            self.indices_it = iter(self.indices_it)

            self.epochs_completed += 1
            self.index_in_epoch = 0
            i = next(self.indices_it)

        return np.load(self.in_data_path + str(i) + ".npy"), np.load(self.out_data_path + str(i) + ".npy")

    def next_mem(self):
        try:
            i = next(self.indices_it)
            self.index_in_epoch += 1

        except StopIteration:
            #The list function performs a shallow copy
            self.indices_it = list(self.indices)

            #works in place
            random.shuffle(self.indices_it)
            self.indices_it = iter(self.indices_it)

            self.epochs_completed += 1
            self.index_in_epoch = 0
            i = next(self.indices_it)

        return self.in_data[i],self.out_data[i]

    def get_batch(self, size=32):
        batchIn = np.empty([size, self.nb_freq, self.spec_length, 1])
        batchOut = np.empty([size, self.nb_freq, self.spec_length, 1])

        for i in range(0,size):
            sample = self.next_mem()
            batchIn[i, :, :, :] = sample[0][:self.nb_freq,:,:]
            batchOut[i, :, :, :] = sample[1][:self.nb_freq,:,:]

        return batchIn, batchOut


    def normalise_divmax(self, samples):

        normalised = samples / max(samples)

        return normalised
