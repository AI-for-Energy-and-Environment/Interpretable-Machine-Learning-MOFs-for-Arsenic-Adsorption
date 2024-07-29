import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys

df = pd.read_csv('MFsconv.csv', usecols=[3])

# Morgan
df['Mol'] = df['organ_smiles'].apply(Chem.MolFromSmiles)
df1 = df['Mol'] 
fps1 = np.array([AllChem.GetMorganFingerprintAsBitVect(x, 2, nBits=1024) for x in df1])
FPs1 = pd.DataFrame(fps1)
FPs1.to_csv('.../Morgan.csv')

# MACCS
df['Mol'] = df['organ_smiles'].apply(Chem.MolFromSmiles)
df2 = df['Mol'] 
fps2 = np.array([MACCSkeys.GenMACCSKeys(x) for x in df2])
FPs2 = pd.DataFrame(fps2)
FPs2.to_csv('.../MACCS.csv')