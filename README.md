This is final project for CS 598 DLH course by:
- Deep Shah (deepss2@illinois.edu)
- Sandeep yerramsetty (sy40@illinois.edu)

## Paper Citation
Gehrmann, S., Dernoncourt, F., Li, Y., Carlson, E. T., Wu, J. T., Welt, J., ... & Celi, L. A. (2018). Comparing deep learning and concept extraction based methods for patient phenotyping from clinical narratives. PloS one, 13(2), e0192360. https://dx.plos.org/10.1371/journal.pone.0192360

## Original source by author in Lua + Python
https://github.com/sebastianGehrmann/phenotyping

# Dependency
## Machine used
* Macbook Pro
* RAM: 32GB
* Processor: 2.2 GHz 6-Core Intel Core i7

## Python dependency to run code
* Python 3.8.8
* pytorch 2.0.0
* gensim 3.8.3
* pandas 1.2.4
* nltk 3.8.1

Command to install the packages
```
pip install <package_name>==<version> [--force-reinstall]
```

# Data download instruction
We are using MIMIC-III dataset (Johnson, A., Pollard, T., & Mark, R. (2019). MIMIC-III Clinical Database Demo (version 1.4). PhysioNet. https://doi.org/10.13026/C2HM2Q.)

* Follow the instruction provided here: https://mimic.mit.edu/docs/gettingstarted/
* You need to become credential user given the sensitivity of MIMIC-III data, once you have done the steps listed above you can go to https://physionet.org/content/mimiciii/1.4/ to download the file containing clinical notes (https://mimic.mit.edu/docs/iii/tables/noteevents/).

# Preprocess
* Since all the code assumes that that the MIMIC-III data will be stored in MIMIC-III.csv, you can either rename your file or update the python file to match your data path.
* Download the annotation.csv which is labeled data.
## Word2Vec
Word2Vec output is input to all the models except baseline, run the following command.
```
python3.8 word2vec.py
```
This will dump the learnt word2vec embedding for the token in `mimiciii_word2vec.wordvectors`

# Training model
## Baseline model
```
python3.8 baseline.py
```
It will output all the result in `baseline/all_class.txt` including various n-gram width.

## CNN model
```
python3.8 CNN_model.py
```
* It will output the result in `cnn_result/<phenotype_class>.txt`.
* You can run different width by changing passing the desired conv_width value to method `train_model_for_phenotype`.
* There are two models in the file `CNNModelDropPostPool` & `CNNModelPoolPostDrop`, you can change the `model = <model_name>` in `train_model_for_phenotype`.

## LSTM model
```
python3.8 LSTM.py
```
* It will output the result in `lstm_result/<phenotype_class>.txt`.
* There are two models in the file `LSTMMaxPoolModel` & `LSTMModel`, you can change the `model = <model_name>` in `train_model_for_phenotype`.

## Pretrained models
You can download the pretrained embedding from the [Google drive](https://drive.google.com/file/d/1ANATL85zoVVoOnTQ5VFvOpYY9-Z102Xg/view)

# Table of Result
Phenotypes | N-Gram Best | CNN Width 1 | CNN Width 1-2
------------ | ------------ | ------------ | ------------
Advanced.Cancer | 91 | 93 | 96
Advanced.Heart.Disease | 87 | 90 | 93
Advanced.Lung.Disease | 90 | 80 | 86
Alcohol.Abuse | 85 | 90 | 93
Chronic.Neurological.Dystrophies | 70 | 76 | 79
Chronic.Pain.Fibromyalgia | 71 | 72 | 76
Depression | 79 | 93 | 95
Obesity | 75 | 74 | 84
Other.Substance.Abuse | 83 | 87 | 88
Schizophrenia.and.other.Psychiatric.Disorders | 75 | 90 | 89
