# Molecules Generation

![alt text](https://github.com/chennnnnyize/Generative-Molecules/blob/master/Graphs/Model_%20Schematic.png)

## User: 

A user preferably has some basic knowledge about command line inputs that can specify filenames, and other conditions.


## Description:

For a given molecules input data (.txt/.csv file with molecular SMILES string, and properties: logP, QED, and SAS), the code can firstly encode the molecular SMILES into a code vector representation, this multidimensional continuous representation can be used to train the GAN model, and GAN will be jointly trained with property prediction to help shape the latent space. The new latent space can then be optimized upon to find the molecules with the most optimized properties of interest.



## Precondition: 

*	Python 
	* Tensorflow
	* RDKit
	* numpy
	* matplotlib
	* pandas
	* argparse
	* progressbar


## Methods

### Molecules Representing
In models for natural language processing, the input and output of the model are usually sequences of single letters, strings or words. We therefore employ the SMILES format, which encodes molecular graphs compactly as human-readable strings. 

Our SMILES-based text encoding used a subset of 35 different characters for ZINC. For ease of computation, we encoded strings up to a maximum length of 120 characters for ZINC. Shorter strings were padded with spaces to this same length. We used only canonicalized SMILES for training to avoid dealing with equivalent SMILES representations. 


### Property Prediction
To discovere new molecules and chemicals in relation to maximizing some desirable property, we extended the purely generative model to also predict property values from the latent representation. We trained a multilayer perceptron jointly with the encoder to predict properties from the latent representation of each molecule.

With joint training for property prediction, the distribution of molecules in the latent space is organized by property values.
The interest in discovering new molecules and chemicals is most often in relation to maximizing some desirable property. For the algorithm
trained on the ZINC dataset, the objective properties include logP, QED, SAS. 


### GAN Training



## Input:

* Molecular Structure: represented in SMILES string
                       
		       eg. aspirin_smiles = 'CC(=O)Oc1ccccc1C(=O)O'

* Molecular Properties: 
	
		Water−octanol partition coefficient (logP);
	
		synthetic accessibility score (SAS);
	
		Quantitative Estimation of Drug-likeness (QED)


## Output:

* SMILES strings
* Molecules visualisation

![alt text](https://github.com/chennnnnyize/Generative-Molecules/blob/master/Graphs/Molecules_Graph.png) 






