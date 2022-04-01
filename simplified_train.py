# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 18:44:04 2019

@author: jacqu

Graph2Smiles VAE training (RGCN encoder, GRU decoder, teacher forced decoding). 

To resume training form a given 
- iteration saved
- learning rate
- beta 

pass corresponding args + load_model = True


"""

import argparse
import sys, os
import torch
import numpy as np

import torch.utils.data
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler

import torch.nn.utils.clip_grad as clip
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from selfies import decoder
from multiprocessing import Pool
script_dir = os.path.dirname(os.path.realpath(__file__))
if __name__=='__main__':
    sys.path.append(script_dir)


from utils import ModelDumper, disable_rdkit_logging, setup, log_reconstruction
from dgl_utils import send_graph_to_device
from model_simplified import Model
from loss_func import VAELoss #, weightedPropsLoss, affsRegLoss, affsClassifLoss
from dataloaders.molDataset_simplified import molDataset, Loader

from rdkit import Chem
def ring_check(smile,ring_size=8):
    smile_length=len(smile)
    molecule_status=0
    ring_status=0
    if smile_length>0 :
        m= Chem.MolFromSmiles(smile)
        molecule_status=1
        for atom in m.GetAtoms():
            i=atom.GetIdx()
            temp=m.GetAtomWithIdx(i).IsInRingSize(ring_size)
            if temp:
                ring_status=1
                break
            else:
                ring_status=0
    return molecule_status,ring_status

def basic_stat(smiles):
    smiles_length=[]
    for smile in smiles:
        smiles_length.append(len(smile))
    smiles_min= min(smiles_length)
    smiles_max=max(smiles_length)
    smiles_mean=np.mean(smiles_length)
    smiles_median=np.median(smiles_length)
    smiles_FQ=np.quantile(smiles_length,0.25)
    smiles_TQ=np.quantile(smiles_length,0.75)
    return smiles_min,smiles_max,smiles_mean,smiles_median,smiles_FQ,smiles_TQ
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', type=str, default='default') # model name in results/saved_models/
    parser.add_argument('--train', help="path to training dataframe", type=str, default='data/moses_train.csv')
    parser.add_argument("--cutoff", help="Max number of molecules to use. Set to -1 for all in csv", type=int, default=-1)
    
    # Alphabets params 
    parser.add_argument('--decode', type=str, default='selfies')  # language used : 'smiles' or 'selfies'
    parser.add_argument('--alphabet_name', type=str, default='moses_alphabets.json') # name of alphabets json file, in map_files dir 
    parser.add_argument('--build_alphabet', action='store_true')  # use params.json alphabet

    # If we start from a pretrained model : 
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_name', type=str, default='default')  # name of model to load from
    parser.add_argument('--load_iter', type=int, default=0)  # resume training at optimize step n°


    # Model architecture 
    parser.add_argument('--decoder_type', type=str, default='GRU')  # name of model to load from
    parser.add_argument('--n_gcn_layers', type=int, default=3)  # number of gcn encoder layers (3 or 4?)
    parser.add_argument('--n_gru_layers', type=int, default=3)  # number of gcn encoder layers (3 or 4?)
    parser.add_argument('--gcn_dropout', type=float, default=0.2)
    parser.add_argument('--gcn_hdim', type=int, default=32)
    parser.add_argument('--latent_size', type=int, default=56) # jtvae uses 56
    parser.add_argument('--gru_hdim', type=int, default=450)
    parser.add_argument('--gru_dropout', type=float, default=0.2)
    
    parser.add_argument('--use_batchNorm', action='store_true') # default uses batchnorm tobe coherent with before 

    # Training schedule params :

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)  # Initial learning rate
    parser.add_argument('--anneal_rate', type=float, default=0.9)  # Learning rate annealing
    parser.add_argument('--anneal_iter', type=int, default=40000)  # update learning rate every _ step
    parser.add_argument('--clip_norm', type=float, default=50.0)  # Gradient clipping max norm
    
    # Kl weight schedule 
    parser.add_argument('--beta', type=float, default=0.0)  # initial KL annealing weight
    parser.add_argument('--step_beta', type=float, default=0.002)  # beta increase per step
    parser.add_argument('--max_beta', type=float, default=0.5)  # maximum KL annealing weight
    parser.add_argument('--warmup', type=int, default=40000)  # number of steps with only reconstruction loss (beta=0)
    parser.add_argument('--kl_anneal_iter', type=int, default=2000)  # update beta every _ step

    parser.add_argument('--print_iter', type=int, default=1000)  # print loss metrics every _ step
    parser.add_argument('--print_smiles_iter', type=int, default=100)  # print reconstructed smiles every _ step
    parser.add_argument('--save_iter', type=int, default=10000)  # save model weights every _ step

    # teacher forcing rnn schedule
    parser.add_argument('--tf_init', type=float, default=1.0)
    parser.add_argument('--tf_step', type=float, default=0.002)  # step decrease
    parser.add_argument('--tf_end', type=float, default=0)  # final tf frequency
    parser.add_argument('--tf_anneal_iter', type=int, default=1000)  # nbr of iters between each annealing
    parser.add_argument('--tf_warmup', type=int, default=70000)  # nbr of steps at tf_init


    
    parser.add_argument('--processes', type=int, default=12)  # num workers
    parser.add_argument('--nmol', type=int, default=1000)  # number of molecules to sample for quality check per epoch

    # =======

    args, _ = parser.parse_known_args()

    logdir, modeldir = setup(args.name, permissive=True)
    dumper = ModelDumper(dumping_path=os.path.join(modeldir, 'params.json'), argparse=args)


    writer = SummaryWriter(logdir)
    disable_rdkit_logging()  # function from utils to disable rdkit logs

    # Load train set and test set
    loaders = Loader(maps_path='map_files/',
                     csv_path=args.train,
                     vocab=args.decode,
                     build_alphabet=args.build_alphabet,
                     alphabet_name = args.alphabet_name, 
                     n_mols=args.cutoff,
                     num_workers=args.processes,
                     batch_size=args.batch_size)
                    #  props=properties,
                    #  targets=targets)

    train_loader, _, test_loader = loaders.get_data()

    # Model & hparams
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    params = {'features_dim': loaders.dataset.emb_size,  # node embedding dimension
              'num_rels': loaders.num_edge_types,
              'gcn_layers': args.n_gcn_layers,
              'gcn_dropout':args.gcn_dropout,
              'gcn_hdim': args.gcn_hdim,
              'gru_hdim':args.gru_hdim,
              'decoder_type': args.decoder_type,
              'gru_dropout': args.gru_dropout,
              'batchNorm': args.use_batchNorm,
              'l_size': args.latent_size,
              'voc_size': loaders.dataset.n_chars,
              'max_len': loaders.dataset.max_len,
              'device': device,
              'index_to_char': loaders.dataset.index_to_char}

    dumper.dic.update(params)
    dumper.dump()

    model = Model(**params).to(device)

    load_model = args.load_model
    load_path = f'results/saved_models/{args.load_name}/params.json'
    if load_model:
        print(f"Careful, I'm loading {args.load_name} in train.py, line 160")
        weights_path = f'results/saved_models/{args.load_name}/weights.pth'
        model.load_state_dict(torch.load(weights_path))

    print(model)
    map = ('cpu' if device == 'cpu' else None)

    # Optim
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, args.anneal_rate)
    print("> learning rate: %.6f" % scheduler.get_lr()[0])

    # Train & test

    w = torch.Tensor([0.01217708, 0.00948469, 0.00377783, 0.00370055, 0.04868181,
       0.02715935, 0.02685141, 0.01828003, 0.        , 0.        ,
       0.        , 0.        , 0.00883069, 0.00717266, 0.00446925,
       0.01183266, 0.0008677 , 0.00217616, 0.06201966, 0.0210205 ,
       0.18292837, 0.        , 0.02212551, 0.        , 0.        ,
       0.        , 0.        , 0.        , 0.08719869, 0.00380422,
       0.08321414, 0.05577309, 0.2959594 , 0.00049454]).to(device)

    if args.load_model:
        total_steps = args.load_iter
    else:
        total_steps = 0
    beta = args.beta
    tf_proba = args.tf_init

    for epoch in range(1, args.epochs + 1):
        print(f'Starting epoch {epoch}')
        model.train()
        epoch_train_rec, epoch_train_kl = 0, 0 
 
        for batch_idx, (graph, smiles) in enumerate(train_loader):

            total_steps += 1  # count training steps

            smiles = smiles.to(device)
            graph = send_graph_to_device(graph, device)


            # Forward passs
            mu, logv, _, out_smi= model(graph, smiles, tf=tf_proba)
            

            # Compute loss terms : change according to multitask setting
            rec, kl = VAELoss(out_smi, smiles, mu, logv, w)



            # COMPOSE TOTAL LOSS TO BACKWARD
            if total_steps < args.warmup:  # Only reconstruction (warmup)
                t_loss = rec
            else:
                t_loss = rec + beta * kl
                
            optimizer.zero_grad()
            t_loss.backward()
            del t_loss
            clip.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            # Annealing KL and LR
            if total_steps % args.anneal_iter == 0:
                scheduler.step()
                print("learning rate: %.6f" % scheduler.get_lr()[0])

            if total_steps % args.kl_anneal_iter == 0 and total_steps >= args.warmup:
                beta = min(args.max_beta, beta + args.step_beta)

            if total_steps % args.tf_anneal_iter == 0 and total_steps >= args.tf_warmup:
                tf_proba = max(args.tf_end, tf_proba - args.tf_step)  # tf decrease

            # logs and monitoring
            if total_steps % args.print_iter == 0:
                print(
                    f'Opt step {total_steps}, rec: {rec.item():.2f}, kl: {beta * kl.item():.2f}')

                writer.add_scalar('BatchRec/train', rec.item(), total_steps)
                writer.add_scalar('BatchKL/train', kl.item(), total_steps)


            if args.print_smiles_iter > 0 and total_steps % args.print_smiles_iter == 0:
                _, out_chars = torch.max(out_smi.detach(), dim=1)
                _, frac_valid = log_reconstruction(smiles, out_smi.detach(),
                                                   loaders.dataset.index_to_char,
                                                   string_type=args.decode)
                print(f'{frac_valid} valid smiles in batch')
                # Correctly reconstructed characters
                differences = 1. - torch.abs(out_chars - smiles)
                differences = torch.clamp(differences, min=0., max=1.).double()
                quality = 100. * torch.mean(differences)
                quality = quality.detach().cpu()
                writer.add_scalar('quality/train', quality.item(), total_steps)
                print('fraction of correct characters at reconstruction : ', quality.item())

            if total_steps % args.save_iter == 0:
                model.cpu()
                torch.save(model.state_dict(), os.path.join(modeldir, "weights.pth"))
                model.to(device)

            # keep track of epoch loss
            epoch_train_rec += rec.item()
            epoch_train_kl += kl.item()

        # Validation pass
        model.eval()
        val_rec, val_kl = 0, 0
        with torch.no_grad():
            for batch_idx, (graph, smiles) in enumerate(test_loader):

                smiles = smiles.to(device)
                graph = send_graph_to_device(graph, device)
                mu, logv, z, out_smi = model(graph, smiles, tf=tf_proba)
                rec, kl = VAELoss(out_smi, smiles, mu, logv,w)

                val_rec += rec.item()
                val_kl += kl.item()


                # Correctly reconstructed characters in first validation batch
                if batch_idx == 0:
                    _, out_chars = torch.max(out_smi.detach(), dim=1)
                    differences = 1. - torch.abs(out_chars - smiles)
                    differences = torch.clamp(differences, min=0., max=1.).double()
                    quality = 100. * torch.mean(differences)
                    quality = quality.detach().cpu()
                    writer.add_scalar('quality/valid', quality.item(), epoch)
                    print('fraction of correct characters in first valid batch : ', quality.item())

            # total Epoch losses
            val_rec, val_kl = val_rec / len(test_loader), val_kl / len(test_loader)
            epoch_train_rec, epoch__train_kl = epoch_train_rec / len(train_loader), epoch_train_kl / len(train_loader)

            print(f'[Ep {epoch}/{args.epochs}], batch valid. loss: rec: {val_rec:.2f}, kl: {beta * kl.item():.2f}')
            # eval molecule quality 
            z = model.sample_z_prior(args.nmol)
            gen_seq = model.decode(z)
            selfies = model.probas_to_smiles(gen_seq)
            smiles = [decoder(s.replace('[PADDING]',''), bilocal_ring_function=True) for s in selfies]
            smiles_min,smiles_max,smiles_mean,smiles_median,smiles_FQ,smiles_TQ= basic_stat(smiles=smiles)
            pool=Pool(args.processes)
            results_check=np.array(pool.map(ring_check,smiles))
            exists, aberant = results_check.sum(axis=0)
            perc_aberrant_mol=aberant/exists
            perc_correct=exists/args.nmol
            perc_unique=len(set(smiles))/args.nmol

        
        
        print(f'[Ep {epoch}/{args.epochs}],smile size stat: min:{smiles_min:.2f}, '
              f'max:{smiles_max:.2f}, mean:{smiles_mean:.2f},median:{smiles_median:.2f}, '
              f'first quartile={smiles_FQ:.2f}, third quartile:{smiles_TQ}')
        
        print(f'[Ep {epoch}/{args.epochs}], aberrant ring smiles fraction:{perc_aberrant_mol:.2f}, '
              f'correct smiles fraction:{perc_correct:.2f} unique smiles fraction:{perc_unique:.2f}')

        print(f'[Ep {epoch}/{args.epochs}], batch valid. loss: rec: {val_rec:.2f}, kl: {beta * kl.item():.2f}')

        # Tensorboard logging
        writer.add_scalar('EpochRec/valid', val_rec, epoch)
        writer.add_scalar('EpochRec/train', epoch_train_rec, epoch)
        writer.add_scalar('EpochKL/valid', val_kl, epoch)
        writer.add_scalar('EpochKL/train', epoch_train_kl, epoch)
        # Tensorboard logging for smiles quality check
        writer.add_scalar('Smiles_metrics/min', smiles_min, epoch)
        writer.add_scalar('Smiles_metrics/max', smiles_max, epoch)
        writer.add_scalar('Smiles_metrics/mean', smiles_mean, epoch)
        writer.add_scalar('Smiles_metrics/median', smiles_median, epoch)
        writer.add_scalar('Smiles_metrics/first_quartile', smiles_FQ, epoch)
        writer.add_scalar('Smiles_metrics/third_quartile', smiles_TQ, epoch)
        # Tensorboard logging for moles quality check
        writer.add_scalar('Molecules_metrics/aberrant_ring', perc_aberrant_mol, epoch)
        writer.add_scalar('Molecules_metrics/correct_molecules', perc_correct, epoch)
        writer.add_scalar('Molecules_metrics/uniqueness', perc_unique, epoch)



