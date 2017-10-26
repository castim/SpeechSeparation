import os, glob
import subprocess
from os.path import basename

audio_dir = "/Users/danielm/Desktop/ductapeguy"
output_dir = "/Users/danielm/Desktop/uct"

# Difference in the speech signal levels in dB.
# Assumes that both audio files have been correctly normalized and have the same speech signal level initially.
spl_difference = 0

speaker_dirs = [os.path.join(audio_dir,fn) for fn in next(os.walk(audio_dir))[2]]
speaker_dirs = glob.glob(os.path.join(audio_dir, '*'))
print(speaker_dirs)

if spl_difference == 0:

    spk1_index = 0
    for speaker1_dir in speaker_dirs:

        speaker1_audios = glob.glob(os.path.join(speaker1_dir, '*.wav'))

        for audio1 in speaker1_audios:

            for speaker2_dir in speaker_dirs[spk1_index:]:
                speaker2_audios = glob.glob(os.path.join(speaker2_dir, '*.wav'))

                for audio2 in speaker2_audios:

                   outFileName = os.path.splitext(basename(speaker1_dir))[0] + "_" + os.path.splitext(basename(audio1))[0] + \
                              "_" + os.path.splitext(basename(speaker2_dir))[0] + os.path.splitext(basename(audio2))[0] + ".wav"
                   outFilePath = os.path.join(output_dir, outFileName)
                   subprocess.call(["sox", "-m", audio1, audio2, outFilePath])
        spk1_index = spk1_index + 1

else:

    spk1_index = 0
    for speaker1_dir in speaker_dirs:

        speaker1_audios = glob.glob(os.path.join(speaker1_dir, '*.wav'))

        for audio1 in speaker1_audios:

            temp_filename = "spk1_temp.wav"
            temp_file_path = os.path.join(output_dir, temp_filename)
            subprocess.call(["sox", audio1, temp_file_path, "gain", "-" +str(spl_difference)])

            for speaker2_dir in (speaker_dirs[:spk1_index] + speaker_dirs[(spk1_index+1):]):
                print(speaker2_dir)
                speaker2_audios = glob.glob(os.path.join(speaker2_dir, '*.wav'))

                for audio2 in speaker2_audios:

                    outFileName = os.path.splitext(basename(speaker1_dir))[0] + "_" + os.path.splitext(basename(audio1))[0] + \
                                  "_" +os.path.splitext(basename(speaker2_dir))[0] + os.path.splitext(basename(audio2))[0] +".wav"
                    outFilePath = os.path.join(output_dir, outFileName)
                    subprocess.call(["sox", "-m",temp_file_path , audio2, outFilePath])
        spk1_index = spk1_index +1
