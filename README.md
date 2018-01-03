# Speech Separation for Hearing aids

A demonstration is available in the Demo.ipynb jupyter notebook.

* rnn_conv.ipynb is a notebook with the model, it's training and it's testing.
* rnn_conv_training.py is a script to train the model.
* build_dataset.py is a script to build the tfRecords dataset from the audios of the LibriSpeech Corpus
* Other files are tools for the first ones

Some files are taken from the Stephenson et al. paper (https://github.com/lab41/magnolia).
These are the files classifying LibriSpeech Corpus
into male and female files in a computer friendly way (the classification comes from the LibriSpeech corpus)
Note that there was a mistake in the classification that was corrected (in the dev set)

The computation of the indices was taken from https://github.com/craffel/mir_eval.
