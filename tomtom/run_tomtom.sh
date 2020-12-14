#!/bin/bash

# run python script to convert tensor to text and save
python pwms_tensor_to_txt.py $1

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
rm pwms.txt
rm $1_tmp.tsv