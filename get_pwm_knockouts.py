# script to assess contribution of each layer 1 conv filter by setting it to its mean output
# and measuring the change in response

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
from get_pwm import get_pwm_model, get_knockout_model
    
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
    
    pwm_model, _ = get_pwm_model()
    n_batch = math.ceil(X_test.shape[0] / batch_size)

    # get prediction outputs from knockout models
    print("Getting predictions from each knockout DeepSEA model...")
    ypreds = torch.load(args.out_dir + "ypred_test.pt")
    deepsea = kipoi.get_model("DeepSEA/predict").model
    n_filters = 320
    kerrs = {'l1': np.zeros((n_filters, ypreds.shape[1])), 
             'l2': np.zeros((n_filters, ypreds.shape[1]))}
    for f in tqdm(range(n_filters)):    
        # get knockout model
        knockout_model = get_knockout_model(deepsea, f)
        errs = {'l1': np.zeros(ypreds.shape[1]), 
                'l2': np.zeros(ypreds.shape[1])}
        
        for batch in tqdm(range(n_batch)):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, X_test.shape[0])
            # make knockout predictions
            kpreds = knockout_model(X_test[start_idx:end_idx, :])
            # compute difference in predictions and error metrics
            resid = kpreds - ypreds[start_idx:end_idx, :]
            errs['l1'] += abs(resid).sum(axis = 0).detach().numpy()
            errs['l2'] += resid.pow(2).sum(axis = 0).detach().numpy()
            
        kerrs['l1'][f, :] = errs['l1'] / ypreds.shape[0]
        kerrs['l2'][f, :] = errs['l2'] / ypreds.shape[0]
        np.save(args.out_dir + 'knockout_errs_l1.npy', kerrs['l1'])
        np.save(args.out_dir + 'knockout_errs_l2.npy', kerrs['l2'])
    
    print("Job completed.")
    

    
    
    
