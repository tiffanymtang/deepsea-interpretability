# script to compute some relevant/helper statistics from the first convolutional layer

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
    
if __name__ == '__main__':
    import scipy.io
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--out_dir', dest = 'out_dir', required = False, default = 'out/')
    parser.add_argument('--data_dir', dest = 'data_dir', required = False, default = 'data/deepsea_train/')
    parser.add_argument('--batch_size', dest = 'batch_size', required = False, default = 500, type = int)
    
    args = parser.parse_args()    
    batch_size = args.batch_size
    
    print("Loading data...")
    test_mat = scipy.io.loadmat(args.data_dir + 'test.mat')
    X_test = torch.FloatTensor(test_mat['testxdata'])
    X_test = torch.reshape(X_test, (455024, 4, 1, 1000))
    
    # reorder to A, C, G, T format
    x = X_test[:, 1:2, :, :].clone()
    y = X_test[:, 2:3, :, :].clone()
    X_test[:, 1:2, :, :] = y
    X_test[:, 2:3, :, :] = x
    
    pwm_model, _ = get_pwm_model()
    n_batch = math.ceil(X_test.shape[0] / batch_size)
    
    # get max output for each sequence and filter from 1st conv + activation layer model
    print("Getting max output for each sequence and filter from layer 1...")
    out_val_ls = []
    out_pos_ls = []
    out_idx_ls = []
    for batch in tqdm(range(n_batch)):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, X_test.shape[0])
        out_val, out_pos = pwm_model(X_test[start_idx:end_idx, :])[:, :, 0, :].max(axis = 2)
        out_val_ls.append(out_val)
        out_pos_ls.append(out_pos)
        out_idx_ls.append(list(range(start_idx, end_idx)) + [x * -1 for x in range(start_idx, end_idx)])
    out_val = torch.cat(out_val_ls)
    out_pos = torch.cat(out_pos_ls)
    out_idx = list(chain.from_iterable(out_idx_ls))
    
    torch.save(out_val, args.out_dir + "max_output_values.pt")
    torch.save(out_pos, args.out_dir + "max_output_positions.pt")
    torch.save(out_idx, args.out_dir + "max_output_indices.npy")

    # get mean output for each filter in 1st conv + activation layer model
    print("Getting mean output for each filter from layer 1...")
    out_sum = torch.zeros(320, 993)
    freq = 0
    for batch in tqdm(range(n_batch)):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, X_test.shape[0])
        out = pwm_model(X_test[start_idx:end_idx, :])
        out_sum = out_sum + out[:, :, 0, :].sum(axis = 0)
        freq += out.shape[0]
    out_mean_seq = out_sum / freq
    torch.save(out_mean_seq, args.out_dir + "mean_sequence_output_layer1.pt")
    torch.save(out_mean_seq.mean(axis = 1), args.out_dir + "mean_output_layer1.pt")
    
    print("Job completed.")
    

    
    
    
