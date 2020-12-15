import gzip
import csv
import random
import numpy as np

# util functions are implemented based on this repository
# https://github.com/MedChaabane/DeepBind-with-PyTorch/blob/master/Binding_sites_Prediction_PyTorch_colab.ipynb
class Chip():
    def __init__(self,filename, motiflen=24,reverse_complemet_mode=True):
        self.file = filename
        self.motiflen = motiflen
        self.reverse_complemet_mode = reverse_complemet_mode
            
    def openFile(self):
        train_dataset = []
        print(self.file)
        with gzip.open(self.file, 'rt') as data:
            next(data)
            reader = csv.reader(data, delimiter = '\t')
            if not self.reverse_complemet_mode:
                for row in reader:
                    train_dataset.append([seqtopad(row[2], self.motiflen), [1]])
                    train_dataset.append([seqtopad(dinucshuffle(row[2]), self.motiflen), [0]])
            else:
                for row in reader:
                    train_dataset.append([seqtopad(row[2], self.motiflen),[1]])
                    train_dataset.append([seqtopad(reverse_complement(row[2]), self.motiflen), [1]])
                    train_dataset.append([seqtopad(dinucshuffle(row[2]), self.motiflen), [0]])
                    train_dataset.append([seqtopad(dinucshuffle(reverse_complement(row[2])), self.motiflen), [0]])
        #random.shuffle(train_dataset)
        train_dataset_pad = train_dataset

        size = int(len(train_dataset_pad) / 3)
        firstvalid = train_dataset_pad[:size]
        secondvalid = train_dataset_pad[size:size + size]
        thirdvalid = train_dataset_pad[size + size:]
        firsttrain = secondvalid + thirdvalid
        secondtrain = firstvalid + thirdvalid
        thirdtrain = firstvalid + secondvalid
        return firsttrain, firstvalid, secondtrain, secondvalid, thirdtrain, thirdvalid, train_dataset_pad


def seqtopad(sequence, motlen):
    rows = len(sequence) #+2*motlen-2
    S = np.empty([rows, 4])
    base = 'ACGT'
    for i in range(rows):
        for j in range(4):
            #if i-motlen+1<len(sequence) and sequence[i-motlen+1]=='N' or i<motlen-1 or i>len(sequence)+motlen-2:
            #    S[i,j]=np.float32(0.25)
            #elif sequence[i-motlen+1]==base[j]:
            if sequence[i] == base[j]:    
                S[i,j] = np.float32(1)
            else:
                S[i,j] = np.float32(0)
    return np.transpose(S)

def dinucshuffle(sequence):
    b = [sequence[i:i + 2] for i in range(0, len(sequence), 2)]
    random.shuffle(b)
    d = ''.join([str(x) for x in b])
    return d

def complement(seq):
    complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N'}
    complseq = [complement[base] for base in seq]
    return complseq
  
def reverse_complement(seq):
    seq = list(seq)
    seq.reverse()
    return ''.join(complement(seq))
    

