import numpy as np
import torch
import sys
import re 

# load PWMs or PFMs in tensor format
pwms = torch.load("../out/" + sys.argv[1] + ".pt")

# check if these are the PWMs direct and convert to between 
# 0 and 1
if re.search("direct", sys.argv[1]) or re.search("filter3", sys.argv[1]): 
    pwms_tmp = np.zeros((pwms.shape[0], pwms.shape[1], pwms.shape[2]))
    for i in range(pwms.shape[0]):
        for j in range(pwms.shape[2]):
            col = pwms[i, :, j].numpy()
            pwms_tmp[i, :, j] = np.abs(
                col - col.min()) / np.abs((col - col.min())).sum()
    file_name = 'pwms.txt'
    file = open(file_name, 'a')
    for i in range(pwms_tmp.shape[0]):
        np.savetxt(file, pwms_tmp[i, :, :])
        file.write("\n")
    file.close()
else: 
    # flip the second and third rows to A C T G format from A G C T
    x = pwms[:, 1:2, :].clone()
    y = pwms[:, 2:3, :].clone()
    pwms[:, 1:2, :] = y
    pwms[:, 2:3, :] = x
    
    file_name = 'pwms.txt'
    file = open(file_name, 'a')
    for i in range(pwms.shape[0]):
        np.savetxt(file, pwms[i, :, :])
        file.write("\n")
    file.close()

