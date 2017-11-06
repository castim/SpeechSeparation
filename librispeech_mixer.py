import os, glob
import subprocess
from os.path import basename
import numpy as np
from scipy.io.wavfile import read
from scipy.signal import spectrogram
import soundfile as sf
import matplotlib.pyplot as plt
from itertools import cycle
import random

class LibriSpeechMixer:

    #If you want to avoid scratching the ssd
    #sudo mount -t tmpfs -o size=500m tmpfs tmpMixed
    output_dir = "tmpMixed/"

    male_audios = []
    female_audios = []

    indices = []
    indices_it = None
    epochs_completed = 0
    index_in_epoch = 0

    #Do we need to implement a equalization?
    spl_difference = 0

    #The length of the spectrogram we take
    spec_length = 200


    # Difference in the speech signal levels in dB.
    # Assumes that both audio files have been correctly normalized and have the same speech signal level initially.

    def __init__(self, train=True):
        if train:
            audio_dir = "Data/LibriSpeech/train-clean-100/"

            female_file = "magnolia/data/librispeech/authors/train-clean-100-F.txt"
            male_file = "magnolia/data/librispeech/authors/train-clean-100-M.txt"
        else:
            audio_dir = "Data/LibriSpeech/dev-clean/"

            female_file = "magnolia/data/librispeech/authors/dev-clean-F.txt"
            male_file = "magnolia/data/librispeech/authors/dev-clean-M.txt"

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

        self.indices = range(0, 100)#min(len(self.male_audios), len(self.female_audios)))

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

        """outFileName = os.path.splitext(basename(self.male_audios[i]))[0] + "_" \
                        + os.path.splitext(basename(self.female_audios[i]))[0] + ".wav"
                        """
        outFilePath = os.path.join(self.output_dir, "temp.wav")
        subprocess.call(["sox", "-m", self.male_audios[i], self.female_audios[i], outFilePath])

        #input is in flac
        #(sample rate is always the same)
        target1, samplerate = sf.read(self.male_audios[i])
        target2, samplerate = sf.read(self.female_audios[i])

        length = min(len(target1), len(target2))

        Pxx_target1, freqs_target1, bins_target1, im_target1 = plt.specgram(target1[:length], Fs=samplerate)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        #plt.show()



        Pxx_target2, freqs_target2, bins_target2, im_target2 = plt.specgram(target2[:length], Fs=samplerate)
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        #plt.show()

        #output is in wav format
        samplerate, mixed = read(outFilePath)

        Pxx_mixed, freqs_mixed, bins_mixed, im_mixed = plt.specgram(mixed[:length], Fs=samplerate)

        """plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')"""
        #plt.show()

        return np.moveaxis(np.array([Pxx_mixed])[:,:,:self.spec_length], 0, -1), \
                            np.moveaxis(np.array([Pxx_target1, Pxx_target2])[:,:,:self.spec_length], 0, -1)


    def get_batch(self, size=32):
        batchIn = np.empty([size, 129, self.spec_length, 1])
        batchOut = np.empty([size, 129, self.spec_length, 2])

        for i in range(0,size):
            sample = self.next()
            batchIn[i, :, :, :] = sample[0]
            batchOut[i, :, :, :] = sample[1]

        return batchIn, batchOut
