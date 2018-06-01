# Generative-Molecules

![alt text](https://github.com/chennnnnyize/Generative-Molecules/blob/master/Graphs/GAN.png)


This is the repo for DIRECT capstone project: Molecules Design Using Deep Generative Models.

Group Members: Xiaoxiao Jia, Jiaxu Qin, Yize Chen

![alt text](https://github.com/chennnnnyize/Generative-Molecules/blob/master/data/examples.png)

## Objective

In this project, we propose to take advantage of recent advances in the research of generative models in machine learning area. An automatic machine learning pipeline for designing and generating new molecules will be developed. We will design an efficient way to convert discrete representations of molecules to a multidimensional continuous representation. This method allows us to approximate sampling valid structures from the distribution of input molecules, and to generate new molecules for efficient exploration and optimization through open-ended spaces of chemical compounds. Existing chemical structures will then be learned and used as the knowledge representations for exploring new molecular structures. 

![111](https://user-images.githubusercontent.com/35084836/40514598-8a799834-5f5e-11e8-8647-c4cac523c314.png)

![222](https://user-images.githubusercontent.com/35084836/40515765-ed355bd0-5f62-11e8-9860-b877cc0ad4da.png)

## Algorithm

### Generative Adversarial Networks(GANs)

The main idea behind a GAN is to have two competing neural network models. One takes noise as input and generates samples ( called the generator, G). The other model (called the discriminator, D) receives samples from both the generator and the training data, and has to be able to distinguish between the two sources. These two networks play a continuous game, where the generator is learning to produce more and more realistic samples, and the discriminator is learning to get better and better at distinguishing generated data from real data. These two networks are trained simultaneously, and the hope is that the competition will drive the generated samples to be indistinguishable from real data. During training, the loss for G and D are shown as follows respectively:

![alt text](https://github.com/chennnnnyize/Generative-Molecules/blob/master/data/equation.png)

which makes use of Wasserstein distance instead of the KL divergence in original GAN.

We would try to incorporate LSTM/GRU units in our neural networks to fulfill the task of generating sequential structures.


## Get the Data
The GAN model was trained on the SMILES file from the ZINC and ChEMBL database, which contains molecules and measured biological activity data. 

A small 50k molecule dataset (`data/smiles_50k.h5`), a much larger 500k ChEMBL 21 extract (`data/smiles_500k.h5`) are included in this repo.

A model trained on `smiles_500k.h5` is included in `data/model_500k.h5`.

**All h5 files in this repo by [git-lfs](https://git-lfs.github.com/) rather than included directly in the repo.**

To download the original datasets (zinc12 and chembl22), you can use the `download_dataset.py` script:

 * `python download_dataset.py --dataset zinc12`
 * `python download_dataset.py --dataset chembl22`


## Preprocess the data

Before training the network, `preprocess.py` is needed to convert SMILES strings into matrix and then output as specified file. The detailed functions of preprocess.py are:

* Normalizes the length of each string to 120 by appending whitespace as needed.
* Builds a list of the unique characters used in the dataset. (The "charset")
* Substitutes each character in each SMILES string with the integer ID of its location in the charset.
* Converts each character position to a one-hot vector of len(charset).
* Saves this matrix to the specified output file.

Example: 

      python preprocess.py data/smiles_50k.h5 data/processed.h5


## Train the network

The preprocessed data can be fed into the `train.py` script:

      python train.py data/processed.h5 model.h5 --epochs 20`



## Results

Waiting to be completed


## Contact
