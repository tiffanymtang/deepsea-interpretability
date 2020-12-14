import numpy as np
import torch
import sys

# load PWMs or PFMs in tensor format
pwms = torch.load("../out/" + sys.argv[1] + ".pt")

# loop through all 919 features 
sum_of_weights = pwms.numpy().sum(axis = (1, 2, 3))
for i in range(pwms.shape[0]):
    if sum_of_weights[i] == 0:
        continue
    file_name = 'pwms_tmp/pwms_' + str(i) + ".txt"
    file = open(file_name, 'a')
    for j in range(pwms.shape[1]):
        np.savetxt(file, pwms[i, j, :, :])
        file.write("\n")
    file.close()

