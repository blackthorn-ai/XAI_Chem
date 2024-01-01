from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs

def smiles_to_fingerprint(smiles, radius=2, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
        return fingerprint
    else:
        return None

def calculate_tanimoto_similarity(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)

training_set = ["CCO", "CCN", "CCC"]

# Створення фінгерпринтів для всіх молекул у навчальному наборі
fingerprints = [smiles_to_fingerprint(smiles) for smiles in training_set if smiles_to_fingerprint(smiles) is not None]

# Обчислення Tanimoto similarity всередині підмножини
for i in range(len(fingerprints)):
    for j in range(i+1, len(fingerprints)):
        similarity = calculate_tanimoto_similarity(fingerprints[i], fingerprints[j])
        print(f"Tanimoto Similarity between {training_set[i]} and {training_set[j]}: {similarity}")
