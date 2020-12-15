# script to get all DeepSEA predictions on test set

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
import os
    
if __name__ == '__main__':
    import scipy.io
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--out_dir', dest = 'out_dir', required = False, default = '../out/')
    parser.add_argument('--data_dir', dest = 'data_dir', required = False, default = '../data/deepsea_train/')
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
    
    deepsea = kipoi.get_model("DeepSEA/predict").model
    n_batch = math.ceil(X_test.shape[0] / batch_size)
    
    # make predictions
    print("Making test predictions...")
    for batch in tqdm(range(n_batch)):
        start_idx = batch * batch_size
        end_idx = min((batch + 1) * batch_size, X_test.shape[0])
        preds = deepsea(X_test[start_idx:end_idx, :])
        torch.save(preds, args.out_dir + "preds" + str(batch) + ".pt")
    
    # combine predictions
    preds_ls = []
    for batch in tqdm(range(n_batch)):
        preds_ls.append(torch.load(args.out_dir + "preds" + str(batch) + ".pt"))
        os.remove(args.out_dir + "preds" + str(batch) + ".pt")
    preds = torch.cat(preds_ls)
    torch.save(preds, args.out_dir + "ypred_test.pt")
    
    print("Job completed.")
    

    
    
    
