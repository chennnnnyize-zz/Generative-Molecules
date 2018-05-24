# Generative-Molecules


This is the repo for DIRECT capstone project: Molecules Design Using Deep Generative Models.

Group Members: Xiaoxiao Jia, Jiaxu Qin, Yize Chen

![alt text](https://github.com/chennnnnyize/Generative-Molecules/blob/master/data/examples.png)

## Objective

In this project, we propose to take advantage of recent advances in the research of generative models in machine learning area. An automatic machine learning pipeline for designing and generating new molecules will be developed. We will design an efficient way to convert discrete representations of molecules to a multidimensional continuous representation. This method allows us to approximate sampling valid structures from the distribution of input molecules, and to generate new molecules for efficient exploration and optimization through open-ended spaces of chemical compounds. Existing chemical structures will then be learned and used as the knowledge representations for exploring new molecular structures. 

![111](https://user-images.githubusercontent.com/35084836/40514598-8a799834-5f5e-11e8-8647-c4cac523c314.png)

## Algorithm

### Generative Adversarial Networks(GANs)
Generative adversarial networks is firstly proposed by Ian Goodfellow et al in 2014, which has been a very successful attempt in generative model. This model makes use of zero-sum game to train two deep neural networks, $G$ the generator $Z->G(Z)$, and $D$ the discriminator $D(X)$ or $D(G(Z))$, simultaneously. During training, the loss for $G$ and $D$ are shown as follows respectively:

![alt text](https://github.com/chennnnnyize/Generative-Molecules/blob/master/data/equation.png)

which makes use of Wasserstein distance instead of the KL divergence in original GAN.

We would try to incorporate LSTM/GRU units in our neural networks to fulfill the task of generating sequential structures.

## Results

Waiting to be completed

## Contact


## Get the Data

A small 50k molecule dataset (data/smiles_50k.h5), a much larger 500k ChEMBL 21 extract (data/smiles_500k.h5), and a model trained on smiles_500k.h5 (data/model_500k.h5) are included.
All h5 files in this repo by git-lfs rather than included directly in the repo.
To download the original datasets (zinc12 and chembl22), you can use the `download_dataset.py` script:

* python download_dataset.py --dataset zinc12
* python download_dataset.py --dataset chembl22


## Preprocess the data

Before training the network, preprocess.py is needed to convert SMILES strings into matrix and then output as specified file. The detailed functions of preprocess.py are:
* Normalizes the length of each string to 120 by appending whitespace as needed.
* Builds a list of the unique characters used in the dataset. (The "charset")
* Substitutes each character in each SMILES string with the integer ID of its location in the charset.
* Converts each character position to a one-hot vector of len(charset).
* Saves this matrix to the specified output file.

Example: 

python preprocess.py data/smiles_50k.h5 data/processed.h5

## Training the network
