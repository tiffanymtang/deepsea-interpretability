# main script to extract pwms from first convolutional+activation layer

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

def get_pwm_model(verbose = False):
    '''
    Get subset of DeepSEA model used to construct PWMs from input sequences
    
    Returns
    -------
    pwm_model : 1st convolutional layer + activation stage model
    rc_model : revese complement data set model
    '''
    
    # load in model
    deep_sea = kipoi.get_model("DeepSEA/predict")
    
    # freeze all weights in pre-trained model
    for name, param in deep_sea.model.named_parameters():
        param.requires_grad = False
        if verbose:
            print('name: ', name)
            print(type(param))
            print('param.shape: ', param.shape)
            print('param.requires_grad: ', param.requires_grad)
            print('=====')
    
    pwm_model = nn.Sequential(*list(deep_sea.model[0:2]), 
                              nn.Sequential(*deep_sea.model[2][0:2]))
    
    rc_model = nn.Sequential(*list(deep_sea.model[0:2]))
    
    return pwm_model, rc_model


def get_knockout_model(full_model, i, verbose = False):
    '''
    Get full DeepSEA model where we knock out one filter in the first convolutional
    layer by setting it to the mean output
    
    Parameters
    ----------
    full_model : full deep sea model; kipoi.get_model("DeepSEA/predict").model
    i : integer; which filter to "knockout"
    verbose : boolean; level of print statements
    
    Returns
    -------
    knockout_model : full deepsea model with one layer1 filter knocked out
    '''
    
    # freeze all weights in pre-trained model
    for name, param in full_model.named_parameters():
        param.requires_grad = False
        if verbose:
            print('name: ', name)
            print(type(param))
            print('param.shape: ', param.shape)
            print('param.requires_grad: ', param.requires_grad)
            print('=====')
    
    mean_layer1_output = torch.load("../out/mean_output_layer1.pt")
    
    knockout_model = copy.deepcopy(full_model)
    knockout_model[2][0].bias[i] = nn.Parameter(mean_layer1_output[i])
    knockout_model[2][0].weight[i, :] = nn.Parameter(torch.zeros_like(full_model[2][0].weight[i, :]))
    
    return knockout_model


def get_pwm(X, batch_size = 500, thr = 1e-6, keep_all_active = True, direct = False, 
            parallel = True, n_cores = 4):
    '''
    Get PWM from input sequences X that pass activation from first conv layer
    
    Parameters
    ----------
    X : torch.Tensor object of size (N, 4, 1, 1000); model input
    batch_size : integer; number of samples in each batch
    thr : float; activation threshold
    keep_all_active : boolean; whether or not to take all active subsequences
    direct : boolean; whether or not to get pwms directly from kernel weights
    parallel : boolean; whether or not to parallelize code
    n_cores : integer; number of cores for parallelization
    
    Returns
    -------
    pwms : torch.Tensor object of size (320, 4, 8); PWMs
    pfms : torch.Tensor object of size (320, 4, 8); PFMs (if direct = False)
    '''
    
    print("Loading in DeepSEA model...")
    pwm_model, rc_model = get_pwm_model()
    
    if direct:
        print("Getting PWMs directly from trained DeepSEA model...")
        pwms = pwm_model[2][0].weight[:, :, 0, :]
        # get a,c,t,g ordering of filters
        x = pwms[:, 1:2, :].clone()
        y = pwms[:, 2:3, :].clone()
        pwms[:, 1:2, :] = y
        pwms[:, 2:3, :] = x
        pfms = None
    else:
        print("Getting PWMs from DeepBind method...")
        
        # run code in batches due to memory constraints
        n_batch = math.ceil(X.shape[0] / batch_size)
        n_filters = 320
        pfms = torch.zeros(n_filters, 4, 8)  # pwm for each filter
        freqs = np.zeros(n_filters)
        
        if parallel:  # run in parallel
            keys = []
            for i in tqdm(range(math.ceil(n_batch / n_cores))):  # run in groups of n_cores
                manager = multiprocessing.Manager()
                return_dict = manager.dict()
                jobs = []
                for batch in range(i * n_cores, min((i + 1) * n_cores, n_batch)):
                    start_idx = batch * batch_size
                    end_idx = min((batch + 1) * batch_size, X.shape[0])                
                    p = multiprocessing.Process(target = get_pfm_batch_parallel, 
                                                args = (X[start_idx:end_idx, :, :, :],
                                                        pwm_model, rc_model, 
                                                        return_dict, batch,
                                                        thr, keep_all_active))
                    jobs.append(p)
                    p.start()
                for p in jobs:
                    p.join()
                pfms = pfms + torch.stack([return_dict[batch]['pfms'] for batch in return_dict.keys()]).sum(axis = 0)
                freqs = freqs + list(np.stack([return_dict[batch]['freqs'] for batch in return_dict.keys()]).sum(axis = 0))
                keys.append(return_dict.keys())
                
            # some error checking
            keys = list(chain.from_iterable(keys))
            if len(keys) != n_batch:
                print("Possible missing batches. Saving keys to file...")
                with open('../out/keys.txt', 'w') as filehandle:
                    for key in keys:
                        filehandle.write('%s\n' % key)
        else:
            for batch in tqdm(range(n_batch)):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, X.shape[0])
                pfms_batch, freqs_batch = get_pfm_batch(X[start_idx:end_idx, :, :, :],
                                                        pwm_model, rc_model, thr, keep_all_active)
                pfms = pfms + pfms_batch
                freqs = freqs + freqs_batch

        # normalize position frequency matrices by frequencies
        pwms = torch.zeros_like(pfms)
        for f in range(n_filters):
            pwms[f, :, :] = pfms[f, :, :] / freqs[f]
    
    return pwms, pfms


def get_pfm_batch(X_batch, pwm_model, rc_model, thr = 1e-6, keep_all_active = True):
    
    '''
    Get PFM from batch of input sequences X that pass activation from first conv layer
    
    Parameters
    ----------
    X : torch.Tensor object of size (N, 4, 1, 1000); model input
    pwm_model: pytorch model to pass sequences through activation threshold
    rc_model: pytorch model to get reverse complement data
    thr : float; activation threshold
    keep_all_active : boolean; whether or not to take all active subsequences
    
    Returns
    -------
    pfms : torch.Tensor object of size (320, 4, 8); position frequency matrix
    freqs: numpy array of size 320 with # of active sequences for each filter
    '''
    
    # feed sequences through truncated model
    print("Feeding sequences through DeepSEA model...")
    out = pwm_model(X_batch)  # output is (2*N, 320, 1, 993)
    rc_data = rc_model(X_batch)  # output is (2*N, 4, 1, 1000)
    
    # create PWM from active segments
    print("Creating PFM from active segments...")
    pfms = torch.zeros(out.shape[1], 4, 8)  # pfm for each filter
    freqs = np.zeros(out.shape[1])  # number of active sequences for each filter
    if keep_all_active:
        # get all segments that pass threshold (default is 1e-6)
        active_idx = out > thr
        if thr < 0:  # use adaptive filter-specific threshold (filter_max_value / 2)
            thrs = torch.load("../out/max_output_values.pt").max(axis = 0).values.numpy() / 2
            for f in range(out.shape[1]):  
                active_idx[:, f, :, :] = out[:, f, :, :] > thrs[f]
        print("Proportion of non-zeros:", (active_idx * 1.0).mean().item())
        
        for f in range(out.shape[1]):  # for each filter
            for idx in range(out.shape[3]):  # for each location in sequence
                active_seg_idx = active_idx[:, f, 0, idx]
                active_seg = rc_data[active_seg_idx, :, 0, idx:(idx + 8)]
                pfms[f, :, :] = pfms[f, :, :] + active_seg.sum(axis = 0)
                freqs[f] += active_seg.shape[0]
    else:
        # get samples that have at least one position that passes threshold
        out_max_value, out_max_pos = out[:, :, 0, :].max(axis = 2)
        active_idx = out_max_value > thr
        if thr < 0:  # use adaptive filter-specific threshold (filter_max_value / 2)
            thrs = torch.load("../out/max_output_values.pt").max(axis = 0).values.numpy() / 2
            for f in range(out.shape[1]):  
                active_idx[:, f] = out_max_value[:, f] > thrs[f]
        out_max_pos[~active_idx] = -1
        print("Proportion of active sequences:", (active_idx * 1.0).mean().item())
        for f in range(out.shape[1]):  # for each filter
            for pos in out_max_pos[:, f].unique():
                if pos == -1:
                    continue
                active_seg_idx = out_max_pos[:, f] == pos
                active_seg = rc_data[active_seg_idx, :, 0, pos:(pos + 8)]
                pfms[f, :, :] = pfms[f, :, :] + active_seg.sum(axis = 0)
                freqs[f] += active_seg.shape[0]
            
    return pfms, freqs


def get_pfm_batch_parallel(X_batch, pwm_model, rc_model, return_dict, batch,
                           thr = 1e-6, keep_all_active = True):
    
    '''
    Get PFM from batch of input sequences X that pass activation from first conv layer;
    for parallelization
    
    Parameters
    ----------
    X : torch.Tensor object of size (N, 4, 1, 1000); model input
    pwm_model: pytorch model to pass sequences through activation threshold
    rc_model: pytorch model to get reverse complement data
    return_dict : dictionary of results to return in multiprocessing.Process()
    batch : key/index to store results in return_dict
    thr : float; activation threshold
    keep_all_active : boolean; whether or not to take all active subsequences
    
    Returns
    -------
    return_dict : a dictionary with two elements
        pfms : torch.Tensor object of size (320, 4, 8); position frequency matrix
        freqs: numpy array of size 320 with # of active sequences for each filter
    '''
    
    # feed sequences through truncated model
    print("Feeding sequences through DeepSEA model...")
    out = pwm_model(X_batch)  # output is (2*N, 320, 1, 993)
    rc_data = rc_model(X_batch)  # output is (2*N, 4, 1, 1000)
    
    # create PWM from active segments
    print("Creating PFM from active segments...")
    pfms = torch.zeros(out.shape[1], 4, 8)  # pfm for each filter
    freqs = np.zeros(out.shape[1])  # number of active sequences for each filter
    if keep_all_active:
        # get all segments that pass threshold (default is 1e-6)
        active_idx = out > thr
        if thr < 0:  # use adaptive filter-specific threshold (filter_max_value / 2)
            thrs = torch.load("../out/max_output_values.pt").max(axis = 0).values.numpy() / 2
            for f in range(out.shape[1]):  
                active_idx[:, f, :, :] = out[:, f, :, :] > thrs[f]
        print("Proportion of non-zeros:", (active_idx * 1.0).mean().item())
        
        for f in range(out.shape[1]):  # for each filter
            for idx in range(out.shape[3]):  # for each location in sequence
                active_seg_idx = active_idx[:, f, 0, idx]
                active_seg = rc_data[active_seg_idx, :, 0, idx:(idx + 8)]
                pfms[f, :, :] = pfms[f, :, :] + active_seg.sum(axis = 0)
                freqs[f] += active_seg.shape[0]
    else:
        # get samples that have at least one position that passes threshold
        out_max_value, out_max_pos = out[:, :, 0, :].max(axis = 2)
        active_idx = out_max_value > thr
        if thr < 0:  # use adaptive filter-specific threshold (filter_max_value / 2)
            thrs = torch.load("../out/max_output_values.pt").max(axis = 0).values.numpy() / 2
            for f in range(out.shape[1]):  
                active_idx[:, f] = out_max_value[:, f] > thrs[f]
        out_max_pos[~active_idx] = -1
        print("Proportion of active sequences:", (active_idx * 1.0).mean().item())
        for f in range(out.shape[1]):  # for each filter
            for pos in out_max_pos[:, f].unique():
                if pos == -1:
                    continue
                active_seg_idx = out_max_pos[:, f] == pos
                active_seg = rc_data[active_seg_idx, :, 0, pos:(pos + 8)]
                pfms[f, :, :] = pfms[f, :, :] + active_seg.sum(axis = 0)
                freqs[f] += active_seg.shape[0]
    
    return_dict[batch] = {"pfms": pfms, "freqs": freqs}
    
    return return_dict
    
    
if __name__ == '__main__':
    import scipy.io
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--out_dir', dest = 'out_dir', required = False, default = 'out/')
    parser.add_argument('--out_tag', dest = 'out_tag', required = False, default = 'all_active_per_seq')
    parser.add_argument('--data_dir', dest = 'data_dir', required = False, default = 'data/deepsea_train/')
    
    parser.add_argument('--batch_size', dest = 'batch_size', required = False, default = 500, type = int)
    parser.add_argument('--thr', dest = 'thr', required = False, default = 1e-6, type = float)
    parser.add_argument('--keep_all_active', dest = 'keep_all_active', action = "store_true")
    parser.add_argument('--direct', dest = 'direct', action = "store_true")
    
    parser.add_argument('--parallel', dest = 'parallel', action = "store_true")
    parser.add_argument('--n_cores', dest = 'n_cores', required = False, default = 3, type = int)
    
    parser.add_argument('--include_y', dest = 'include_y', required = False, default = None)
    parser.add_argument('--observed_y', dest = 'observed_y', required = False, default = 1, type = int)
    parser.add_argument('--quantile_y', dest = 'quantile_y', required = False, default = .9, type = float)
    
    args = parser.parse_args()    
    
    test_mat = scipy.io.loadmat(args.data_dir + 'test.mat')
    X_test = torch.FloatTensor(test_mat['testxdata'])
    X_test = torch.reshape(X_test, (455024, 4, 1, 1000))
        
    # reorder to A, C, G, T format
    x = X_test[:, 1:2, :, :].clone()
    y = X_test[:, 2:3, :, :].clone()
    X_test[:, 1:2, :, :] = y
    X_test[:, 2:3, :, :] = x
        
    if args.include_y is None:  # perform analysis with no knowledge of y or yhat
        print("Arguments:")
        print(args)
        pwms, pfms = get_pwm(X_test,
                             batch_size = args.batch_size, 
                             thr = args.thr, 
                             keep_all_active = args.keep_all_active,
                             direct = args.direct, 
                             parallel = args.parallel,
                             n_cores = args.n_cores)
    else:  # perform analysis with respect to particular columns in y or yhat
        ynames = pd.read_csv(args.data_dir + 'predictor.names', header = None)
        y_keep_idx = np.where(ynames.iloc[:, 0].str.contains("CTCF|PTBP1|SOX10|TP53|POU2F2|HNRNPA1|PRDM1|EBF1|NR4A2|ZC3H10").values)[0]
        if args.include_y == 'observed':  # look only at y == 1 sequences
            Y_test = torch.FloatTensor(test_mat['testdata'])
            keep_idx = Y_test == args.observed_y
            args.out_tag = args.out_tag + '_yobserved_' + str(args.observed_y)
        elif args.include_y == 'predicted':  # look only at yhat "high" sequences
            Y_test = torch.load(args.out_dir + 'ypred_test.pt')
            keep_idx = torch.BoolTensor(Y_test.shape)
            for i in range(Y_test.shape[1]):
                keep_idx[:, i] = Y_test[:, i] > np.quantile(Y_test[:, i].detach().numpy(),
                                                            args.quantile_y)
            args.out_tag = args.out_tag + '_ypredicted_' + str(args.quantile_y)

        print("Arguments:")
        print(args)
        
        pwms = torch.zeros(Y_test.shape[1], 320, 4, 8)
        pfms = torch.zeros(Y_test.shape[1], 320, 4, 8)
        for i in y_keep_idx:
            if keep_idx[:, i].sum() < 100:
                continue
            print(str(i) + ": " + ynames.iloc[i, 0])
            pwms[i, :], pfms[i, :] = get_pwm(X_test[keep_idx[:, i], :], 
                                             batch_size = args.batch_size, 
                                             thr = args.thr, 
                                             keep_all_active = args.keep_all_active,
                                             direct = args.direct, 
                                             parallel = args.parallel,
                                             n_cores = args.n_cores)
            torch.save(pwms, args.out_dir + "PWMs_" + args.out_tag + '.pt')
            torch.save(pfms, args.out_dir + "PFMs_" + args.out_tag + '.pt')
            
    torch.save(pwms, args.out_dir + "PWMs_" + args.out_tag + '.pt')
    torch.save(pfms, args.out_dir + "PFMs_" + args.out_tag + '.pt')

    print("Job completed.")
    

    
    
    
