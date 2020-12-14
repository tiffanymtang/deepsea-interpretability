#!/bin/bash

# run meme script to convert to meme file 
cat pwms.txt | matrix2meme > pwms.meme

# run tomtom algorithm and save results 
tomtom -text pwms.meme -thresh $2 ../data/JASPAR-2020/* > $1_tmp.tsv

# remove the last three rows 
head -n-3 $1_tmp.tsv | cat > $1.tsv

# move to results file 
mv $1.tsv results

# remove text and meme files 
rm pwms.meme 
rm $1_tmp.tsv