#!/bin/bash

# run python script to convert tensor to text and save
mkdir pwms_tmp
python pwms_pred_to_txt.py $1

# make a director to store the files in 
mkdir results/$1 

# loop through the files in pwms_tmp
cd pwms_tmp
for pwm in *
do
	# run meme script to convert to meme file 
    cat $pwm | matrix2meme > pwms.meme

    # run tomtom algorithm and save results 
    tomtom -text pwms.meme ../../data/JASPAR-2020/* > tmp_$pwm.tsv

    # remove the last three rows 
    head -n-3 tmp_$pwm.tsv | cat > $pwm.tsv

    # move to results file 
    mv $pwm.tsv ../results/$1

    # remove text and meme files 
    rm pwms.meme 
    rm tmp_$pwm.tsv
done
cd ..
rm -rf pwms_tmp
