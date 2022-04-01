# -*- coding: utf-8 -*-
"""
Created on Sat Oct 26 18:06:44 2019

@author: jacqu

Dataset class for SMILES to graph 

"""

import os
import sys

script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__ == "__main__":
    sys.path.append(os.path.join(script_dir, '..'))

import torch
import dgl
import pandas as pd
import numpy as np

import pickle
import json
import networkx as nx
from selfies import encoder, decoder
from ordered_set import OrderedSet
from rdkit import Chem

from torch.utils.data import Dataset, DataLoader, Subset
from data_processing.rdkit_to_nx import smiles_to_nx


def collate_block(samples):
    """
    Collates samples into batches. 
    Samples : list of tuples of size 4 
    removes 'None' graphs (reduces batch size)
    """

    samples = [s for s in samples if s[0]!=None]
    
    graphs, smiles = map(list, zip(*samples))
    # graphs, smiles, p_labels, a_labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)

    # p_labels, a_labels = torch.tensor(p_labels), torch.tensor(a_labels)
    smiles = torch.tensor(np.array(smiles), dtype=torch.long)

    return batched_graph, smiles#, p_labels, a_labels


def oh_tensor(category, n): 
    # One-hot float tensor construction
    t = torch.zeros(n, dtype=torch.float)
    t[category] = 1.0
    return t


class molDataset(Dataset):
    """ 
    pytorch Dataset for training on small molecules graphs + smiles 
    """

    def __init__(self, csv_path,
                 maps_path,
                 vocab, 
                 build_alphabet,
                 alphabet_name,
                #  props,
                #  targets,
                 n_mols=-1,
                 graph_only=False, 
                 compute_selfies = False):
        
        self.graph_only=graph_only
        self.compute_selfies = compute_selfies 
        
        # 0/ two options: empty loader or csv path given 
        if csv_path is None:
            print("Empty dataset initialized. Use pass_dataset or pass_dataset_path to add molecules.")
            self.df = None
            self.n = 0

        else:
            # Cutoff number of molecules 
            if n_mols != -1:
                self.df = pd.read_csv(csv_path, nrows=n_mols)
                self.n = n_mols
                print('Dataset columns:', self.df.columns)
            else:
                self.df = pd.read_csv(csv_path)
                self.n = self.df.shape[0]
                print('Dataset columns:', self.df.columns)

        # 1/ ============== Properties & Targets handling: ================

        # self.targets = targets
        # if len(targets)>0:
        #     self.binned_scores = bool('binned' in targets[0]) # whether we use binned affs or true values;
        # else:
        #     self.binned_scores = False
        # self.props = props

        # =========== 2/ Graphs handling ====================

        with open(os.path.join(maps_path, 'edges_and_nodes_map.pickle'), "rb") as f:
            self.edge_map = pickle.load(f)
            self.at_map = pickle.load(f)
            self.chi_map = pickle.load(f)
            self.charges_map = pickle.load(f)

        self.num_edge_types, self.num_atom_types = len(self.edge_map), len(self.at_map)
        self.num_charges, self.num_chir = len(self.charges_map), len(self.chi_map)
        print('> Loaded edge and atoms types to one-hot mappings')
        self.emb_size = 16  # node embedding size = number of node features 

        # 3/ =========== SMILES and SELFIES handling : ================== 
        
        self.language = vocab # smiles or selfies 
        
        if build_alphabet: # Parsing dataset to build custom alphabets
            print('!! Building custom alphabet for dataset. Set build_alphabet=False to use predefined alphabet.')
            selfies_a , len_selfies, smiles_a, len_smiles = self._get_selfie_and_smiles_alphabets()
            if self.language == 'smiles':
                self.max_len = len_smiles
                self.alphabet = smiles_a
                
            elif self.language=='selfies':
                self.max_len = len_selfies
                self.alphabet = selfies_a
            
        else: # params.json alphabets and length
            print(f'-> Careful : Using PREDEFINED selfies alphabet : {alphabet_name}')
            
            with open(os.path.join(maps_path, alphabet_name)) as json_file:
                alphabets_dict = json.load(json_file)

            if self.language == 'smiles':
                self.alphabet = alphabets_dict['smiles_alphabet']
                self.max_len = alphabets_dict['largest_smiles_len']
                
            elif self.language=='selfies':
                self.alphabet = alphabets_dict['selfies_alphabet']
                self.max_len = alphabets_dict['largest_selfies_len']
                
            else:
                print("decode format not understood: 'smiles' or 'selfies' ")
                raise NotImplementedError
        
        self.char_to_index = dict((c, i) for i, c in enumerate(self.alphabet))
        self.index_to_char = dict((i, c) for i, c in enumerate(self.alphabet))
        self.n_chars = len(self.alphabet)

        print(f"> Loaded alphabet. Using {self.language}. Max sequence length allowed is {self.max_len}")


    def pass_dataset_path(self, path, graph_only = True):
        # Pass a new dataset to the loader, without changing other parameters 
        self.df = pd.read_csv(path)
        self.n = self.df.shape[0]
        self.graph_only=graph_only
        #print('New dataset columns:', self.df.columns)

    def pass_dataset(self, df, graph_only = True):
        self.df = df
        self.n = df.shape[0]
        self.graph_only=graph_only
        #print('New dataset columns:', self.df.columns)

    def pass_smiles_list(self, smiles):
        # pass smiles list to the model; a dataframe with unique column 'can' will be created 
        self.df = pd.DataFrame.from_dict({'smiles': smiles})
        self.n = self.df.shape[0]
        self.graph_only=True
        #print('New dataset contains only smiles // no props or affinities')

    def __len__(self):
        return self.n
    
    def _get_selfie_and_smiles_alphabets(self):
        """
        Returns alphabet and length of largest molecule in SMILES and SELFIES, for the training set passed.
            self.df Column's name must be 'smiles'.
        output:
            - selfies alphabet
            - longest selfies string
            - smiles alphabet (character based)
            - longest smiles string
        """
        smiles_list = np.asanyarray(self.df.smiles)
        
        selfies_list = np.asanyarray(self.df.selfies)
        
        smiles_alphabet=list(OrderedSet(''.join(smiles_list)))
        largest_smiles_len=len(max(smiles_list, key=len))
        selfies_len=[]
        
        print(f'--> Building alphabets for {smiles_list.shape[0]} smiles and selfies in dataset...')
        for individual_selfie in selfies_list:
            selfies_len.append(len(individual_selfie)-len(individual_selfie.replace('[',''))) # len of SELFIES
        selfies_alphabet_pre=list(OrderedSet(''.join(selfies_list)[1:-1].split('][')))
        selfies_alphabet=[]
        for selfies_element in selfies_alphabet_pre:
            selfies_alphabet.append('['+selfies_element+']')        
        largest_selfies_len=max(selfies_len)
        
        print('Finished parsing smiles and selfies alphabet. Saving to json file custom_alphabets.json')
        print('Longest selfies : ',  largest_selfies_len)
        print('Longest smiles : ',  largest_smiles_len)
        
        d= {'selfies_alphabet':selfies_alphabet,
            'largest_selfies_len':largest_selfies_len,
            'smiles_alphabet': smiles_alphabet,
            'largest_smiles_len': largest_smiles_len}
        
        with open('custom_alphabets.json', 'w') as outfile:
            json.dump(d, outfile)
        
        return (selfies_alphabet, largest_selfies_len, smiles_alphabet, largest_smiles_len)
    
    
    def selfies_to_hot(self, molecule):
        """
        Go from a single selfies string to a list of integers
        """
        # integer encode input smile
        len_of_molecule=len(molecule)-len(molecule.replace('[',''))
        
        if len_of_molecule >= self.max_len:
            valid_flag = 0
            return 0, valid_flag
            
        for _ in range(self.max_len - len_of_molecule): # selfies padding 
            molecule+='[PADDING]'

        selfies_char_list_pre=molecule[1:-1].split('][')
        selfies_char_list=[]
        for selfies_element in selfies_char_list_pre:
            selfies_char_list.append('['+selfies_element+']')   
    
        try:
            integer_encoded = [self.char_to_index[char] for char in selfies_char_list]
            a = np.array(integer_encoded)
            valid_flag = 1 
        except:
            a = 0
            for char in selfies_char_list :
                if char not in self.char_to_index :
                    print(char)
            valid_flag = 0  # no one hot encoding possible : ignoring molecule 
                
        return a, valid_flag

    def __getitem__(self, idx):
        # Returns tuple 
        # Smiles has to be in first column of the csv !!

        row = self.df.iloc[idx,:]
        
        smiles = row.smiles # needed anyway to build graph 
        m=Chem.MolFromSmiles(smiles)
        
        if self.compute_selfies:
            Chem.Kekulize(m)
            k = Chem.MolToSmiles(m, isomericSmiles=False, kekuleSmiles = True) # kekuleSmiles
            selfie = encoder(k)
            """
            if selfie != row.selfies:
                print('new selfie:', selfie)
                print('prev : ', row.selfies)
            """
            
        else:
            selfie = row.selfies
        
        # 1 - Graph building
        if m!=None:
            graph = smiles_to_nx(smiles)
        else:
            return None, 0,0,0

        one_hot = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                   (nx.get_edge_attributes(graph, 'bond_type')).items()}
        nx.set_edge_attributes(graph, name='one_hot', values=one_hot)

        try:
            at_type = {a: oh_tensor(self.at_map[label], self.num_atom_types) for a, label in
                       (nx.get_node_attributes(graph, 'atomic_num')).items()}
            nx.set_node_attributes(graph, name='atomic_num', values=at_type)
        except KeyError:
            print('!!!! Atom type to one-hot error for input ', smiles, ' ignored')
            return None, 0,0,0

        at_charge = {a: oh_tensor(self.charges_map[label], self.num_charges) for a, label in
                     (nx.get_node_attributes(graph, 'formal_charge')).items()}
        nx.set_node_attributes(graph, name='formal_charge', values=at_charge)
        
        try:
            hydrogens = {a: torch.tensor(self.chi_map[label], dtype=torch.float) for a, label in
                       (nx.get_node_attributes(graph, 'num_explicit_hs')).items()}
            nx.set_node_attributes(graph, name='num_explicit_hs', values=hydrogens)
        except KeyError:
            print('!!!! Number of explicit hydrogens to one-hot error for input ', smiles, ' ignored')
            return None, 0,0,0
        
        aromatic = {a: torch.tensor(self.chi_map[label], dtype=torch.float) for a, label in
                   (nx.get_node_attributes(graph, 'is_aromatic')).items()}
        nx.set_node_attributes(graph, name='is_aromatic', values=aromatic)

        at_chir = {a: torch.tensor(self.chi_map[label], dtype=torch.float) for a, label in
                   (nx.get_node_attributes(graph, 'chiral_tag')).items()}
        nx.set_node_attributes(graph, name='chiral_tag', values=at_chir)

        # to dgl 
        g_dgl = dgl.DGLGraph()
        node_features = ['atomic_num', 'formal_charge', 'num_explicit_hs', 'is_aromatic', 'chiral_tag']
        g_dgl.from_networkx(nx_graph=graph,
                            node_attrs=node_features,
                            edge_attrs=['one_hot']) 
        
        N=g_dgl.number_of_nodes()

        g_dgl.ndata['h'] = torch.cat([g_dgl.ndata[f].view(N,-1) for f in node_features], dim=1)
        
        if self.graph_only: # give only the graph (to encode in latent space)
            return g_dgl, 0,0,0

        # 2 - Smiles / selfies to integer indices array
        
        if self.language == 'selfies':
            
            a, valid_flag = self.selfies_to_hot(selfie)
            if valid_flag ==0 : # no one hot encoding for this selfie, ignore 
                print('!!! Selfie to one-hot failed with current alphabet')
                return None, 0,0,0
            
        else:
            a = np.zeros(self.max_len)
            idces = [self.char_to_index[c] for c in smiles]
            a[:len(idces)] = idces

        # 3 - Optional props and affinities 
        
        # props, targets = 0, 0
        # if len(self.props)>0:
        #     props = np.array(row[self.props], dtype=np.float32)

        # if len(self.targets)>0 and self.binned_scores:
        #         targets = np.array(row[self.targets], dtype=np.int64) # for torch.long class labels
        # elif len(self.targets)>0 :
        #     targets = np.array(row[self.targets], dtype=np.float32) # for torch.float values 

        # targets = np.nan_to_num(targets) # if nan somewhere, change to 0.
            

        return g_dgl, a, # props, targets


class Loader():
    def __init__(self,
                #  props, 
                #  targets,
                 csv_path=None,
                 maps_path ='../map_files/',
                 vocab='selfies',
                 build_alphabet = False,
                 alphabet_name='moses_alphabets.json',
                 n_mols=None,
                 batch_size=64,
                 num_workers=12,
                 graph_only=False, # Only load molecular graph (to get latent embeddings)
                 test_only=False,
                 redo_selfies = False):
        """
        Wrapper for test loader, train loader 
        Uncomment to add validation loader 
        if test_only: puts all molecules in csv in the test loader. Returns empty train and valid loaders

        """

        self.vocab = vocab
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = molDataset(
                                #   props=props,
                                #   targets=targets,
                                  csv_path=csv_path,
                                  maps_path=maps_path, 
                                  vocab=vocab,
                                  build_alphabet = build_alphabet,
                                  alphabet_name=alphabet_name,
                                  n_mols=n_mols,
                                  graph_only=graph_only, 
                                  compute_selfies = redo_selfies)

        self.num_edge_types, self.num_atom_types = self.dataset.num_edge_types, self.dataset.num_atom_types
        self.num_charges = self.dataset.num_charges
        self.test_only = test_only

    def get_maps(self):
        # Returns dataset mapping of edge and node features 
        return self.dataset.edge_map, self.dataset.at_map, self.dataset.chi_map, self.dataset.charges_map

    def get_reverse_maps(self):
        # Returns maps of one-hot index to actual feature 
        rev_em = {v: i for (i, v) in self.dataset.edge_map.items()}
        rev_am = {v: i for (i, v) in self.dataset.at_map.items()}
        rev_chi_m = {v: i for (i, v) in self.dataset.chi_map.items()}
        rev_cm = {v: i for (i, v) in self.dataset.charges_map.items()}
        return rev_em, rev_am, rev_chi_m, rev_cm

    def get_data(self):
        n = len(self.dataset)
        
        indices = list(range(n))
        np.random.shuffle(indices)
        if not self.test_only: # 90% train ; 10 % valid
            split_train, split_valid = 0.95, 0.95
            train_index, valid_index = int(split_train * n), int(split_valid * n)

        else: # all mols in test (use for test time)
            split_train, split_valid = 0, 0
            train_index, valid_index = 0, 0

        train_indices = indices[:train_index]

        test_indices = indices[valid_index:]

        train_set = Subset(self.dataset, train_indices)

        test_set = Subset(self.dataset, test_indices)
        #print(f"Dataset contains {n} samples (train subset: {len(train_set)}, Test subset:{len(test_set)}) ")

        if not self.test_only:
            train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=collate_block, drop_last=True)


        test_loader = DataLoader(dataset=test_set, shuffle=False, batch_size=self.batch_size,
                                 num_workers=self.num_workers, collate_fn=collate_block, drop_last=False)

        # return train_loader, valid_loader, test_loader
        if not self.test_only:
            return train_loader, 0, test_loader
        else:
            return 0, 0, test_loader


if __name__ == '__main__':
    L=Loader(props=[], targets = [], csv_path=None)
