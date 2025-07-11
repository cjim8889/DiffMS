import random
from collections import Counter

import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Descriptors

import multiprocessing
import os

random.seed(42)

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def read_from_sdf(path):
    res = []
    app = False
    with open(path, 'r') as f:
        for line in tqdm(f.readlines(), desc='Loading SDF structures', leave=False):
            if app:
                res.append(line.strip())
                app = False
            if line.startswith('> <SMILES>'):
                app = True

    return res

def filter(mol):
    try:
        smi = Chem.MolToSmiles(mol, isomericSmiles=False) # remove stereochemistry information
        mol = Chem.MolFromSmiles(smi)

        if "." in smi:
            return False
        
        if Descriptors.MolWt(mol) >= 1500:
            return False
        
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() != 0:
                return False
    except:
        return False
    
    return True

FILTER_ATOMS = {'C', 'N', 'S', 'O', 'F', 'Cl', 'H', 'P'}

def filter_with_atom_types(mol):
    try:
        smi = Chem.MolToSmiles(mol, isomericSmiles=False) # remove stereochemistry information
        mol = Chem.MolFromSmiles(smi)

        if "." in smi:
            return False
        
        if Descriptors.MolWt(mol) >= 1500:
            return False
        
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge() != 0:
                return False
            if atom.GetSymbol() not in FILTER_ATOMS:
                return False
    except:
        return False
    
    return True

def process_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        smi = Chem.MolToSmiles(mol, isomericSmiles=False)  # remove stereochemistry information
        mol = Chem.MolFromSmiles(smi)
        if filter_with_atom_types(mol):
            return Chem.MolToInchi(mol)
    except:
        pass
    return None

splits = {'train': 'data/train-00000-of-00001-e9b227f8c7259c8b.parquet', 'validation': 'data/validation-00000-of-00001-9368b7243ba1bff8.parquet'}
df = pd.read_parquet("hf://datasets/sagawa/pubchem-10m-canonicalized/" + splits["train"])

pubchem_train_path = "./data/fp2mol/pubchem/preprocessed/pubchem_train.csv"
pubchem_val_path = "./data/fp2mol/pubchem/preprocessed/pubchem_val.csv"

if os.path.exists(pubchem_train_path) and os.path.exists(pubchem_val_path):
    pubchem_train_df = pd.read_csv(pubchem_train_path)
    pubchem_train_inchis = list(pubchem_train_df["inchi"])
    pubchem_val_df = pd.read_csv(pubchem_val_path)
    pubchem_val_inchis = list(pubchem_val_df["inchi"])
else:
    pubchem_set_raw = set(df["smiles"])

    pubchem_smiles_list = list(pubchem_set_raw)
    with multiprocessing.Pool() as pool:
        results = list(tqdm(pool.imap(process_smiles, pubchem_smiles_list), total=len(pubchem_smiles_list), desc="Cleaning PubChem structures", leave=False))
    pubchem_set = set(r for r in results if r is not None)

    pubchem_inchis = list(pubchem_set)
    random.shuffle(pubchem_inchis)

    pubchem_train_inchis = pubchem_inchis[:int(0.95 * len(pubchem_inchis))]
    pubchem_val_inchis = pubchem_inchis[int(0.95 * len(pubchem_inchis)):]

    pubchem_train_df = pd.DataFrame({"inchi": pubchem_train_inchis})
    pubchem_train_df.to_csv(pubchem_train_path, index=False)

    pubchem_val_df = pd.DataFrame({"inchi": pubchem_val_inchis})
    pubchem_val_df.to_csv(pubchem_val_path, index=False)

excluded_inchis = set(pubchem_val_inchis)

msg_split = pd.read_csv('./data/msg/split.tsv', sep='\t')

msg_labels = pd.read_csv('./data/msg/labels.tsv', sep='\t')
msg_labels["name"] = msg_labels["spec"]
msg_labels = msg_labels[["name", "smiles"]].reset_index(drop=True)

msg_labels = msg_labels.merge(msg_split, on="name")

def process_msg(item):
    smiles, split_val = item
    try:
        mol = Chem.MolFromSmiles(smiles)
        smi = Chem.MolToSmiles(mol, isomericSmiles=False) # remove stereochemistry information
        mol = Chem.MolFromSmiles(smi)
        inchi = Chem.MolToInchi(mol)

        if split_val == "train":
            if filter(mol):
                return (inchi, split_val)
            else:
                return None
        else:
            return (inchi, split_val)
    except:
        return None

msg_items = list(zip(msg_labels["smiles"], msg_labels["split"]))

with multiprocessing.Pool() as pool:
    results = list(tqdm(pool.imap(process_msg, msg_items), total=len(msg_items), desc="Converting MSG SMILES to InChI", leave=False))

msg_train_inchis = []
msg_test_inchis = []
msg_val_inchis = []

for res in results:
    if res is not None:
        inchi, split_val = res
        if split_val == "train":
            msg_train_inchis.append(inchi)
        elif split_val == "test":
            msg_test_inchis.append(inchi)
        elif split_val == "val":
            msg_val_inchis.append(inchi)

msg_train_inchis = list(set(msg_train_inchis))
msg_test_inchis = list(set(msg_test_inchis))
msg_val_inchis = list(set(msg_val_inchis))  

msg_train_df = pd.DataFrame({"inchi": msg_train_inchis})
msg_train_df.to_csv("./data/fp2mol/msg/preprocessed/msg_train.csv", index=False)

msg_test_df = pd.DataFrame({"inchi": msg_test_inchis})
msg_test_df.to_csv("./data/fp2mol/msg/preprocessed/msg_test.csv", index=False)

msg_val_df = pd.DataFrame({"inchi": msg_val_inchis})
msg_val_df.to_csv("./data/fp2mol/msg/preprocessed/msg_val.csv", index=False)

excluded_inchis.update(msg_test_inchis + msg_val_inchis)

combined_inchis = msg_train_inchis + msg_val_inchis + pubchem_train_inchis + pubchem_val_inchis
combined_inchis = list(set(combined_inchis))
random.shuffle(combined_inchis)

combined_train_inchis = combined_inchis[:int(0.95 * len(combined_inchis))]
combined_val_inchis = combined_inchis[int(0.95 * len(combined_inchis)):]
combined_train_inchis = [inchi for inchi in combined_train_inchis if inchi not in excluded_inchis]

combined_train_df = pd.DataFrame({"inchi": combined_train_inchis})
combined_train_df.to_csv("./data/fp2mol/combined/preprocessed/combined_train.csv", index=False)

combined_val_df = pd.DataFrame({"inchi": combined_val_inchis})
combined_val_df.to_csv("./data/fp2mol/combined/preprocessed/combined_val.csv", index=False)