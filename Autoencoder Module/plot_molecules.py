

from rdkit import Chem
from rdkit.Chem import Draw
size = (100, 100)
import matplotlib.pyplot as plt


m = Chem.MolFromSmiles('c1ccc2c(c1)ccn3ccc4c5ccccc5nc4c23')
Draw.MolToFile(m,'mol8.png')
