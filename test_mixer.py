import os, glob
import numpy as np
from scipy.signal import spectrogram, stft, istft
import random
from pydub import AudioSegment
from multiprocessing import Pool

class TestMixer:

    #If you want to avoid scratching the ssd
    #sudo mount -t tmpfs -o size=500m tmpfs tmpMixed
    output_dir = "tmpMixed/"

    #The length of the spectrogram we take
    spec_length = 128
    nb_freq = 128

    # Difference in the speech signal levels in dB.
    # Assumes that both audio files have been correctly normalized and have the same speech signal level initially.

    def __init__(self, nbSamples = float("inf"), nbSpeakers = float("inf"), dataset_built=True):
        self.male_audios = []
        self.female_audios = []
        self.indices = []
        self.epochs_completed = 0
        self.index_in_epoch = 0
        self.epochs_completed_test = 0
        self.index_in_epoch_test = 0
        self.in_file_ind_train = 0
        self.in_file_ind_test = 0

        audio_dir = "Data/LibriSpeech/train-clean-100/"

        female_file = "magnolia/data/librispeech/authors/train-clean-100-F.txt"
        male_file = "magnolia/data/librispeech/authors/train-clean-100-M.txt"

        self.in_data_path_train = "/mnt/train/in/spec"
        self.out_data_path_train = "/mnt/train/out/spec"

        self.in_data_path_test = "/mnt/dev/in/spec"
        self.out_data_path_test = "/mnt/dev/out/spec"

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

        self.sep = round(0.8*maxInd)
        self.indices_train = self.indices[:self.sep]
        self.indices_test = self.indices[self.sep:]

        if not dataset_built:
            self.build_dataset()

        #The list function performs a shallow copy
        self.indices_train_it = list(self.indices_train)
        #works in place
        random.shuffle(self.indices_train_it)
        self.indices_train_it = iter(self.indices_train_it)

        self.current_sample_train = next(self.indices_train_it)
        self.current_sample_train_in = np.load(self.in_data_path_train + str(self.current_sample_train) + '.npy')
        self.current_sample_train_out = np.load(self.out_data_path_train + str(self.current_sample_train) + '.npy')

        self.indices_test_it = [x - self.sep for x in self.indices_test]
        #works in place
        random.shuffle(self.indices_test_it)
        self.indices_test_it = iter(self.indices_test_it)

        self.current_sample_test = next(self.indices_test_it)
        self.current_sample_test_in = np.load(self.in_data_path_test + str(self.current_sample_test) + '.npy')
        self.current_sample_test_out = np.load(self.out_data_path_test + str(self.current_sample_test) + '.npy')

    def mix_and_save(self, i, j, in_path, out_path):
        sound1 = AudioSegment.from_file(self.male_audios[i], format='flac')
        target1 = self.normalise_divmax(np.array(sound1.get_array_of_samples()))

        sound2 = AudioSegment.from_file(self.female_audios[j],format='flac')
        target2 = self.normalise_divmax(np.array(sound2.get_array_of_samples()))

        length = min(len(target1), len(target2))

        freqs_target1, bins_target1, Pxx_target1 = stft(target1[:length])
        freqs_target2, bins_target2, Pxx_target2 = stft(target2[:length])
        mask_target = np.abs(Pxx_target1) / (np.abs(Pxx_target2) + np.abs(Pxx_target1) + 1e-100)

        Fxx_mixed = Pxx_target1 + Pxx_target2

        np.save(in_path + str(i), np.moveaxis(np.array([Fxx_mixed])[:, :self.nb_freq, :], 0, -1))
        np.save(out_path + str(i), np.moveaxis(np.array([mask_target])[:, :self.nb_freq, :], 0, -1))

    def build_dataset(self):
        indices_iterator = list(self.indices_train)
        random.shuffle(indices_iterator)

        p = Pool(9)
        #enumerate to take different females for the males
        p.starmap(self.mix_and_save, [lambda x: (x[0], x[1], self.in_data_path_train, self.out_data_path_train), \
                                                                    enumerate(indices_iterator)])


        indices_iterator = list(self.indices_test)
        random.shuffle(indices_iterator)

        #enumerate to take different females for the males
        p.starmap(self.mix_and_save, [lambda x: (x[0], x[1], self.in_data_path_test, self.out_data_path_test), \
                                                                    enumerate(indices_iterator)])

    def build_dataset_in_mem(self):
        indices_iterator = list(self.indices_train)
        random.shuffle(indices_iterator)

        self.in_data_train = []
        self.out_data_train = []

        #enumerate to take different females for the males
        for i, j in enumerate(indices_iterator):

            sound1 = AudioSegment.from_file(self.male_audios[i], format='flac')
            target1 = self.normalise_divmax(np.array(sound1.get_array_of_samples()))

            sound2 = AudioSegment.from_file(self.female_audios[j],format='flac')
            target2 = self.normalise_divmax(np.array(sound2.get_array_of_samples()))

            length = min(len(target1), len(target2))

            freqs_target1, bins_target1, Pxx_target1 = stft(target1[:length])
            freqs_target2, bins_target2, Pxx_target2 = stft(target2[:length])
            mask_target = np.abs(Pxx_target1) / (np.abs(Pxx_target2) + np.abs(Pxx_target1) + 1e-100)

            Fxx_mixed = Pxx_target1 + Pxx_target2

            self.in_data_train.append(np.moveaxis(np.array([Fxx_mixed])[:, :self.nb_freq, :], 0, -1))
            self.out_data_train.append(np.moveaxis(np.array([mask_target])[:, :self.nb_freq, :], 0, -1))

        indices_iterator = list(self.indices_test)
        random.shuffle(indices_iterator)

        self.in_data_test = []
        self.out_data_test = []

        #enumerate to take different females for the males
        for i, j in enumerate(indices_iterator):

            sound1 = AudioSegment.from_file(self.male_audios[i], format='flac')
            target1 = self.normalise_divmax(np.array(sound1.get_array_of_samples()))

            sound2 = AudioSegment.from_file(self.female_audios[j],format='flac')
            target2 = self.normalise_divmax(np.array(sound2.get_array_of_samples()))

            length = min(len(target1), len(target2))

            freqs_target1, bins_target1, Pxx_target1 = stft(target1[:length])
            freqs_target2, bins_target2, Pxx_target2 = stft(target2[:length])
            mask_target = np.abs(Pxx_target1) / (np.abs(Pxx_target2) + np.abs(Pxx_target1) + 1e-100)

            Fxx_mixed = Pxx_target1 + Pxx_target2

            self.in_data_test.append(np.moveaxis(np.array([Fxx_mixed])[:, :self.nb_freq, :], 0, -1))
            self.out_data_test.append(np.moveaxis(np.array([mask_target])[:, :self.nb_freq, :], 0, -1))

    def next_file_train(self):
        try:
            self.index_in_epoch += 1
            self.in_file_ind_train += self.spec_length
            if self.in_file_ind_train > self.current_sample_train_in.shape[1]:

                self.current_sample_train = next(self.indices_train_it)
                self.current_sample_train_in = np.load(self.in_data_path_train + str(self.current_sample_train) + '.npy')
                self.current_sample_train_out = np.load(self.out_data_path_train + str(self.current_sample_train) + '.npy')
                self.in_file_ind_train = self.spec_length

        except StopIteration:
            #The list function performs a shallow copy
            self.indices_train_it = list(self.indices_train)

            #works in place
            random.shuffle(self.indices_train_it)
            self.indices_train_it = iter(self.indices_train_it)

            self.epochs_completed += 1
            self.index_in_epoch = 0
            self.current_sample_train = next(self.indices_train_it)
            self.current_sample_train_in = np.load(self.in_data_path_train + str(self.current_sample_train) + '.npy')
            self.current_sample_train_out = np.load(self.out_data_path_train + str(self.current_sample_train) + '.npy')
            self.in_file_ind_train = self.spec_length

        return np.abs(self.current_sample_train_in[:,self.in_file_ind_train-self.spec_length:self.in_file_ind_train, :]),\
                self.current_sample_train_out[:,self.in_file_ind_train-self.spec_length:self.in_file_ind_train, :],\
                np.angle(self.current_sample_train_in[:,self.in_file_ind_train-self.spec_length:self.in_file_ind_train, :])

    def next_file_test(self):
        try:
            self.index_in_epoch += 1
            self.in_file_ind_test += self.spec_length
            if self.in_file_ind_test > self.current_sample_test_in.shape[1]:

                self.current_sample_test = next(self.indices_test_it)
                self.current_sample_test_in = np.load(self.in_data_path_test + str(self.current_sample_test) + '.npy')
                self.current_sample_test_out = np.load(self.out_data_path_test + str(self.current_sample_test) + '.npy')
                self.in_file_ind_test = self.spec_length

        except StopIteration:
            self.indices_test_it = [x - self.sep for x in self.indices_test]

            #works in place
            random.shuffle(self.indices_test_it)
            self.indices_test_it = iter(self.indices_test_it)

            self.epochs_completed_test += 1
            self.index_in_epoch_test = 0
            self.current_sample_test = next(self.indices_test_it)
            self.current_sample_test_in = np.load(self.in_data_path_test + str(self.current_sample_test) + '.npy')
            self.current_sample_test_out = np.load(self.out_data_path_test + str(self.current_sample_test) + '.npy')
            self.in_file_ind_test = self.spec_length

        return np.abs(self.current_sample_test_in[:,self.in_file_ind_test-self.spec_length:self.in_file_ind_test, :]),\
                self.current_sample_test_out[:,self.in_file_ind_test-self.spec_length:self.in_file_ind_test, :],\
                np.angle(self.current_sample_test_in[:,self.in_file_ind_test-self.spec_length:self.in_file_ind_test, :])

    def next_mem_train(self):
        try:
            self.index_in_epoch += 1
            self.in_file_ind_train += self.spec_length
            if self.in_file_ind_train > self.in_data_train[self.current_sample_train].shape[1]:

                self.current_sample_train = next(self.indices_train_it)
                self.in_file_ind_train = self.spec_length

        except StopIteration:
            #The list function performs a shallow copy
            self.indices_train_it = list(self.indices_train)

            #works in place
            random.shuffle(self.indices_train_it)
            self.indices_train_it = iter(self.indices_train_it)

            self.epochs_completed += 1
            self.index_in_epoch = 0
            self.current_sample_train = next(self.indices_train_it)
            self.in_file_ind_train = self.spec_length

        return np.abs(self.in_data_train[self.current_sample_train][:,self.in_file_ind_train-self.spec_length:self.in_file_ind_train, :]),\
                self.out_data_train[self.current_sample_train][:,self.in_file_ind_train-self.spec_length:self.in_file_ind_train, :],\
                np.angle(self.in_data_train[self.current_sample_train][:,self.in_file_ind_train-self.spec_length:self.in_file_ind_train, :])

    def next_mem_test(self):
        try:
            self.index_in_epoch += 1
            self.in_file_ind_test += self.spec_length
            if self.in_file_ind_test > self.in_data_test[self.current_sample_test].shape[1]:

                self.current_sample_test = next(self.indices_test_it)
                self.in_file_ind_test = self.spec_length

        except StopIteration:
            self.indices_test_it = [x - self.sep for x in self.indices_test]

            #works in place
            random.shuffle(self.indices_test_it)
            self.indices_test_it = iter(self.indices_test_it)

            self.epochs_completed_test += 1
            self.index_in_epoch_test = 0
            self.current_sample_test = next(self.indices_test_it)
            self.in_file_ind_test = self.spec_length

        return np.abs(self.in_data_test[self.current_sample_test][:,self.in_file_ind_test-self.spec_length:self.in_file_ind_test, :]),\
                self.out_data_test[self.current_sample_test][:,self.in_file_ind_test-self.spec_length:self.in_file_ind_test, :],\
                np.angle(self.in_data_test[self.current_sample_test][:,self.in_file_ind_test-self.spec_length:self.in_file_ind_test, :])

    def get_batch(self, size=32):

        batchIn = np.empty([size, self.nb_freq, self.spec_length, 1])
        batchOut = np.empty([size, self.nb_freq, self.spec_length, 1])
        batchPhase = np.empty([size, self.nb_freq, self.spec_length, 1])

        for i in range(0,size):
            sample = self.next_mem_train()
            batchIn[i, :, :, :] = sample[0][:self.nb_freq,:,:]
            batchOut[i, :, :, :] = sample[1][:self.nb_freq,:,:]
            batchPhase[i, :, :, :] = sample[2][:self.nb_freq, :, :]

        return batchIn, batchOut, batchPhase

    def get_batch_test(self, size=32):

        batchIn = np.empty([size, self.nb_freq, self.spec_length, 1])
        batchOut = np.empty([size, self.nb_freq, self.spec_length, 1])
        batchPhase = np.empty([size, self.nb_freq, self.spec_length, 1])

        for i in range(0,size):
            sample = self.next_mem_test()
            batchIn[i, :, :, :] = sample[0][:self.nb_freq,:,:]
            batchOut[i, :, :, :] = sample[1][:self.nb_freq,:,:]
            batchPhase[i, :, :, :] = sample[2][:self.nb_freq, :, :]

        return batchIn, batchOut, batchPhase

    def normalise_divmax(self, samples):

        normalised = samples / max(samples)

        return normalised
