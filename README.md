# Generative-Molecules


This is the DIRECT capstone project.

Group Members: Xiaoxiao Jia, Jiaxu Qin, Yize Chen

## Objective

In this project, we propose to take advantage of recent advances in the research of generative models in machine learning area. An automatic machine learning pipeline for designing and generating new molecules will be developed. We will design an efficient way to convert discrete representations of molecules to a multidimensional continuous representation. This method allows us to approximate sampling valid structures from the distribution of input molecules, and to generate new molecules for efficient exploration and optimization through open-ended spaces of chemical compounds. Existing chemical structures will then be learned and used as the knowledge representations for exploring new molecular structures. 

## Algorithm


## Results


## Contact


## Get the Data

A small 50k molecule dataset (data/smiles_50k.h5), a much larger 500k ChEMBL 21 extract (data/smiles_500k.h5), and a model trained on smiles_500k.h5 (data/model_500k.h5) are included.
All h5 files in this repo by git-lfs rather than included directly in the repo.
To download the original datasets (zinc12 and chembl22), you can use the `download_dataset.py` script:

* python download_dataset.py --dataset zinc12
* python download_dataset.py --dataset chembl22


## Preprocess the data


## Training the network
