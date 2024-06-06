# MPDL-Enhancer
The repository was developed to identify and characterize enhancers using a multi-perspective deep learning framework (MPDL-Enhancer) 

# Requirements
- python == 3.6

- numpy == 1.19.5

- pytest == 7.0.1

- tensorflow-gpu == 2.6.0

- keras == 2.6.0

- scikit-learn == 0.24.2

- pandas == 1.1.5

- matplotlib == 3.3.4

- shap == 0.39.0

# Usage 

## Create virtual environment with following command: 

>***1)*** conda create env -n MPDL-Enhancer python=3.6.
>
>***2)*** conda activate MPDL-Enhancer.
>
>***3)*** pip install requirements.txt.

## Extracting semantic information using deep learning networks 

>***1)*** Run *dna2vec.py* to get the word embedding matrix of the dna sequence.
>
>***2)*** Run *DL-network.py* to extract semantic information about the enhancers. 

## Extracting structural features of enhancer sequences 

>***1)*** Run DNAshape.R on the R language platform to obtain the shape features of the sequence.
>***2)*** Run *SF-filter.py* to get other sequence structural features and then merge them with shape features for feature filtering by Adaboost.
>

By modifying the --testDataset parameter, you can choose whether to extract features from the training or test set.

## Integrate semantic information and sequence structural features for the final prediction

Run *Prediction.py* to get the prediction results for the sequences.




