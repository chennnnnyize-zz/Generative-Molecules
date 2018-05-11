# Use Cases


## Name: Molecule Generation with Target Properties


## User: 

A user preferably has some basic knowledge about command line inputs that can specify filenames, and other conditions.


## Description:


For a given molecules input data (.txt/.csv file with molecular SMILES string, and properties: logP, QED, and SAS), the code can firstly encode the molecular SMILES into a code vector representation, this multidimensional continuous representation can be used to train the GAN model, and GAN will be jointly trained with property prediction to help shape the latent space. The new latent space can then be optimized upon to find the molecules with the most optimized properties of interest.



## Precondition: 


* Python 3.5

* Tensorflow

* RDKit

* numpy

* matplotlib



## Input:

* Molecular Structure: represented in SMILES string

* Molecular Properties: 
	Water−octanol partition coefficient (logP)
	synthetic accessibility score (SAS)
	Quantitative Estimation of Drug-likeness (QED)


## Output:
* SMILES strings





