import os, glob
import subprocess
from os.path import basename

audio_dir = "Data/LibriSpeech/train-clean-100/"
output_dir = "Data/mixedTrain/"

female_file = "magnolia/data/librispeech/authors/train-clean-100-F.txt"
male_file = "magnolia/data/librispeech/authors/train-clean-100-M.txt"

# Difference in the speech signal levels in dB.
# Assumes that both audio files have been correctly normalized and have the same speech signal level initially.
spl_difference = 0

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

if spl_difference == 0:

    for male_dir in male_speaker_dirs:

        male_audios = glob.glob(os.path.join(male_dir, '*.flac'))

        for audio1 in male_audios:

            for female_dir in female_speaker_dirs:
                female_audios = glob.glob(os.path.join(female_dir, '*.flac'))

                for audio2 in female_audios:
                    print(os.path.splitext(basename(male_dir)))
                    outFileName = os.path.splitext(basename(male_dir))[0] + "_" + os.path.splitext(basename(audio1))[0] + \
                              "_" + os.path.splitext(basename(female_dir))[0] + os.path.splitext(basename(audio2))[0] + ".wav"
                    outFilePath = os.path.join(output_dir, outFileName)
                    subprocess.call(["sox", "-m", audio1, audio2, outFilePath])

else:

    for male_dir in male_speaker_dirs:

        male_audios = glob.glob(os.path.join(male_dir, '*.flac'))

        for audio1 in male_audios:

            temp_filename = "spk1_temp.flac"
            temp_file_path = os.path.join(output_dir, temp_filename)
            subprocess.call(["sox", audio1, temp_file_path, "gain", "-" +str(spl_difference)])

            for female_dir in female_speaker_dirs:
                female_audios = glob.glob(os.path.join(female_dir, '*.flac'))

                for audio2 in female_audios:

                    outFileName = os.path.splitext(basename(male_dir))[0] + "_" + os.path.splitext(basename(audio1))[0] + \
                                  "_" +os.path.splitext(basename(female_dir))[0] + os.path.splitext(basename(audio2))[0] +".wav"
                    outFilePath = os.path.join(output_dir, outFileName)
                    subprocess.call(["sox", "-m",temp_file_path , audio2, outFilePath])
