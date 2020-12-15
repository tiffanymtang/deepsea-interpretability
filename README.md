# Diving into the DeepSEA: From an interpretability point of view

DeepSEA is a multi-task convolutional neural network (CNN) that was shown to accurately predict large-scale chromatin-profiling data, namely TF binding, DNase I sensitivity, and histone-mark profiles [(Zhou and Troyanskaya 2015)](https://www.nature.com/articles/nmeth.3547). We explore various aspects of the CNN to try to understand if the network is learning to detect binding site motifs or more generally, how the CNN is making its predictions. In this repository, we provide code to

- extract position weight matrices (PWMs) from the first convolutional layer as was done in [Alipanahi et al. (2015)](https://www.nature.com/articles/nbt.3300)
- find matches between these PWMs and the JASPAR database [(Sandelin et al. 2004)](https://pubmed.ncbi.nlm.nih.gov/14681366/) using the Tomtom algorithm [(Gupta et al. 2007)](https://pubmed.ncbi.nlm.nih.gov/17324271/)
- evaluate filter importances, that is, which filter is important for which response, using a knockout approach inspired by [Maslova et al. 2019](https://www.biorxiv.org/content/10.1101/2019.12.21.885814v1)
- extract motifs from the learned network using TF-MoDISco [(Shrikumar et al. 2018)](https://arxiv.org/abs/1811.00416v2?utm_source=dlvr.it&utm_medium=twitter)

## Data

Source: http://deepsea.princeton.edu/help/
- DeepSEA training and test data bundle from [here](http://deepsea.princeton.edu/media/code/deepsea_train_bundle.v0.9.tar.gz)
- Names of 919 responses from [here](http://deepsea.princeton.edu/media/code/deepsea.v0.94c.tar.gz) (File location: resources/predictor.names)

## Acknowledgement

Part of our code is adapted from the following repositories:
- [Kipoi python-api tutorial](https://github.com/kipoi/kipoi/blob/master/notebooks/python-api.ipynb)
- [Kipoi-interpret tutorial](https://github.com/kipoi/kipoi-interpret/blob/master/notebooks/1-DNA-seq-model-example.ipynb)
- [DeepBind with Pytorch](https://github.com/MedChaabane/DeepBind-with-PyTorch)
- [TF-MoDSIco tutorial](https://github.com/kundajelab/tfmodisco)
