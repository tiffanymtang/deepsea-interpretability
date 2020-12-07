# script to look at distribution of activations and to look at pwms when splitting
# up bimodal activation distributions into two separate pwms

import numpy as np
import pandas as pd
import kipoi
import torch
import torch.nn as nn
from tqdm import tqdm
import math
import multiprocessing
import copy
from itertools import chain
from get_pwm import get_pwm_model

def get_pwm_bimodal(X, batch_size = 500, f = 2, thr0 = 1e-6, thr1 = 0.3, 
                    keep_all_active = True, parallel = True, n_cores = 4):
    '''
    Get two PWMs from input sequences X that pass activation from first conv layer 
    where activation is divided into two bins
    
    Parameters
    ----------
    X : torch.Tensor object of size (N, 4, 1, 1000); model input
    batch_size : integer; number of samples in each batch
    f : integer; filter id to look at
    thr0 : float; min activation threshold
    thr1 : float; second activation threshold
    keep_all_active : boolean; whether or not to take all active subsequences
    parallel : boolean; whether or not to parallelize code
    n_cores : integer; number of cores for parallelization
    
    Returns
    -------
    pwms0 : torch.Tensor object of size (320, 4, 8); PWMs using sequences between thr0 and thr1
    pfms0 : torch.Tensor object of size (320, 4, 8); PFMs corresponding to pwms0 (if direct = False)
    pwms1 : torch.Tensor object of size (320, 4, 8); PWMs using sequences > thr1
    pfms1 : torch.Tensor object of size (320, 4, 8); PFMs corresponding to pwms1 (if direct = False)
    '''
    
    print("Loading in DeepSEA model...")
    pwm_model, rc_model = get_pwm_model()
    
    print("Getting PWMs from DeepBind method...")

    # run code in batches due to memory constraints
    n_batch = math.ceil(X.shape[0] / batch_size)
    pfms0 = torch.zeros(4, 8)
    pfms1 = torch.zeros(4, 8)
    freqs0 = 0
    freqs1 = 0

    if parallel:  # run in parallel
        keys = []
        for i in tqdm(range(math.ceil(n_batch / n_cores))):  # run in groups of n_cores
            manager = multiprocessing.Manager()
            return_dict = manager.dict()
            jobs = []
            for batch in range(i * n_cores, min((i + 1) * n_cores, n_batch)):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, X.shape[0])                
                p = multiprocessing.Process(target = get_pfm_batch_bimodal_parallel, 
                                            args = (X[start_idx:end_idx, :, :, :],
                                                    f, pwm_model, rc_model, 
                                                    return_dict, batch,
                                                    thr0, thr1, keep_all_active))
                jobs.append(p)
                p.start()
            for p in jobs:
                p.join()
            pfms0 = pfms0 + torch.stack([return_dict[batch]['pfms0'] for batch in return_dict.keys()]).sum(axis = 0)
            freqs0 += np.array([return_dict[batch]['freqs0'] for batch in return_dict.keys()]).sum()
            pfms1 = pfms1 + torch.stack([return_dict[batch]['pfms1'] for batch in return_dict.keys()]).sum(axis = 0)
            freqs1 += np.array([return_dict[batch]['freqs1'] for batch in return_dict.keys()]).sum()
            keys.append(return_dict.keys())

        # some error checking
        keys = list(chain.from_iterable(keys))
        if len(keys) != n_batch:
            print("Possible missing batches. Saving keys to file...")
            with open('out/keys.txt', 'w') as filehandle:
                for key in keys:
                    filehandle.write('%s\n' % key)
    else:
        raise ValueError("Not implemented yet. Set parallel = True.")

    # normalize position frequency matrices by frequencies
    pwms0 = pfms0 / freqs0
    pwms1 = pfms1 / freqs1
    
    return pwms0, pwms1, pfms0, pfms1


def get_pfm_batch_bimodal_parallel(X_batch, f, pwm_model, rc_model, return_dict, batch,
                                   thr0 = 1e-6, thr1 = .3, keep_all_active = True):
    
    '''
    Get PFM from batch of input sequences X that pass activation from first conv layer;
    for parallelization
    
    Parameters
    ----------
    X : torch.Tensor object of size (N, 4, 1, 1000); model input
    f : integer; filter id to look at
    pwm_model: pytorch model to pass sequences through activation threshold
    rc_model: pytorch model to get reverse complement data
    return_dict : dictionary of results to return in multiprocessing.Process()
    batch : key/index to store results in return_dict
    thr0 : float; min activation threshold
    thr1 : float; second activation threshold
    keep_all_active : boolean; whether or not to take all active subsequences
    
    Returns
    -------
    return_dict : a dictionary with two elements
        pfms0 : torch.Tensor object of size (4, 8); position frequency matrix for lower activations
        freqs0 : numpy array of size 1 with # of active sequences for given filter used in pfms0
        pfms1 : torch.Tensor object of size (4, 8); position frequency matrix for higher activations
        freqs1 : numpy array of size 1 with # of active sequences for given filter used in pfms1
    '''
    
    # feed sequences through truncated model
    print("Feeding sequences through DeepSEA model...")
    out = pwm_model(X_batch)[:, f, 0, :]  # output is (2*N, 993)
    rc_data = rc_model(X_batch)  # output is (2*N, 4, 1, 1000)
    
    # create PWM from active segments
    print("Creating PFM from active segments...")
    pfms0 = torch.zeros(4, 8)  # pfm for given filter
    pfms1 = torch.zeros(4, 8)  # pfm for given filter
    freqs0 = 0  # number of active sequences for given filter
    freqs1 = 0  # number of active sequences for given filter
    if keep_all_active:
        # get all segments that pass threshold (default is 1e-6)
        active0_idx = (out > thr0) & (out < thr1)
        active1_idx = out >= thr1

        for idx in range(out.shape[1]):  # for each location in sequence
            
            active0_seg_idx = active0_idx[:, idx]
            active0_seg = rc_data[active0_seg_idx, :, 0, idx:(idx + 8)]
            pfms0 += active0_seg.sum(axis = 0)
            freqs0 += active0_seg.shape[0]
            
            active1_seg_idx = active1_idx[:, idx]
            active1_seg = rc_data[active1_seg_idx, :, 0, idx:(idx + 8)]
            pfms1 += active1_seg.sum(axis = 0)
            freqs1 += active1_seg.shape[0]

    else:
        # get samples that have at least one position that passes threshold
        out_max_value, out_max_pos = out.max(axis = 1)
        active0_idx = (out_max_value > thr0) & (out_max_value < thr1)
        active1_idx = out_max_value >= thr1
        inactive_idx = out_max_value <= thr0
        out_max_pos[inactive_idx] = -1
        for pos in out_max_pos.unique():
            if pos == -1:
                continue
            active0_seg_idx = (out_max_pos == pos) * active0_idx
            active1_seg_idx = (out_max_pos == pos) * active1_idx
            if active0_seg_idx.sum() > 0:
                active0_seg = rc_data[active0_seg_idx, :, 0, pos:(pos + 8)]
                pfms0 += active0_seg.sum(axis = 0)
                freqs0 += active0_seg.shape[0]
            if active1_seg_idx.sum() > 0:
                active1_seg = rc_data[active1_seg_idx, :, 0, pos:(pos + 8)]
                pfms1 += active1_seg.sum(axis = 0)
                freqs1 += active1_seg.shape[0]
    
    return_dict[batch] = {"pfms0": pfms0, "freqs0": freqs0, "pfms1": pfms1, "freqs1": freqs1}
    
    return return_dict
    
    
if __name__ == '__main__':
    import scipy.io
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--out_dir', dest = 'out_dir', required = False, default = 'out/')
    parser.add_argument('--out_tag', dest = 'out_tag', required = False, default = 'all_active_per_seq_bimodal')
    parser.add_argument('--data_dir', dest = 'data_dir', required = False, default = 'data/deepsea_train/')
    
    parser.add_argument('--eval_all', dest = 'eval_all', action = "store_true")
    parser.add_argument('--batch_size', dest = 'batch_size', required = False, default = 500, type = int)
    parser.add_argument('--f', dest = 'f', required = False, default = 2, type = int)
    parser.add_argument('--thr0', dest = 'thr0', required = False, default = 1e-6, type = float)
    parser.add_argument('--thr1', dest = 'thr1', required = False, default = 0.3, type = float)
    parser.add_argument('--keep_all_active', dest = 'keep_all_active', action = "store_true")
    
    parser.add_argument('--parallel', dest = 'parallel', action = "store_true")
    parser.add_argument('--n_cores', dest = 'n_cores', required = False, default = 3, type = int)
    
    args = parser.parse_args()    
    
    print("Loading data...")
    test_mat = scipy.io.loadmat(args.data_dir + 'test.mat')
    X_test = torch.FloatTensor(test_mat['testxdata'])
    X_test = torch.reshape(X_test, (455024, 4, 1, 1000))
        
    # reorder to A, C, G, T format
    x = X_test[:, 1:2, :, :].clone()
    y = X_test[:, 2:3, :, :].clone()
    X_test[:, 1:2, :, :] = y
    X_test[:, 2:3, :, :] = x
    
    # get activations for each sequence
    if eval_all:
        print("Getting activations for each sequence and filter from layer 1...")
        pwm_model, _ = get_pwm_model()
        n_batch = math.ceil(X_test.shape[0] / args.batch_size)
        n_filters = 320
        act_vals_ls = [[] for i in range(n_filters)]
        zero_vals = torch.zeros(n_filters)
        for batch in tqdm(range(300)):  # look only at a subset of 300*batch_size
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, X_test.shape[0])
            act_vals = pwm_model(X_test[start_idx:end_idx, :])[:, :, 0, :]
            zero_vals += (act_vals == 1e-6).sum(axis = [0,2])
            for f in range(n_filters):
                vals = act_vals[:, f, :].flatten()
                vals = vals[vals != 1e-6]
                act_vals_ls[f].append(vals)
        zero_vals = zero_vals / (end_idx * 993 * 2)
        torch.save(zero_vals, args.out_dir + "nonactive_freq.pt")
    
#     # visualizing distribution of first 12 filters
#     import matplotlib.pyplot as plt
#     fig, axs = plt.subplots(4, 3)
#     for i in range(4):
#         for j in range(3):
#             tmp = torch.cat(act_vals_ls[i*4 + j])
#             axs[i, j].hist(tmp, density = 1)
    
    # look at pwms from lower and upper activations using some threshold
    pwms0, pwms1, pfms0, pfms1 = get_pwm_bimodal(X_test, 
                                                 batch_size = args.batch_size, 
                                                 f = args.f,
                                                 thr0 = args.thr0, 
                                                 thr1 = args.thr1, 
                                                 keep_all_active = args.keep_all_active, 
                                                 parallel = args.parallel,
                                                 n_cores = args.n_cores)
    torch.save(pwms0, args.out_dir + "PWMs_" + args.out_tag + str(args.f) + '_0.pt')
    torch.save(pfms0, args.out_dir + "PFMs_" + args.out_tag + str(args.f) + '_0.pt')
    torch.save(pwms1, args.out_dir + "PWMs_" + args.out_tag + str(args.f) + '_1.pt')
    torch.save(pfms1, args.out_dir + "PFMs_" + args.out_tag + str(args.f) + '_1.pt')

    print("Job completed.")