from librispeech_mixer import LibriSpeechMixer

#Create the LibriSpeech mixer
#mixer = TestMixer(dataset_built=False)

mixer = LibriSpeechMixer(dataset_built=False)
validation_mixer = LibriSpeechMixer(train=False, dataset_built=False)
