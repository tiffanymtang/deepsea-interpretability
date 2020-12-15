# get max activation pwms
python get_pwm.py --out_tag="max_active_per_seq" --parallel > ../out/out_files/max_active_per_seq.out

# get max activation pwms
python get_pwm.py --out_tag="all_active_per_seq" --keep_all_active --parallel > ../out/out_files/all_active_per_seq.out

# get pwms directly from first layer
python get_pwm.py --out_tag="direct" --direct > ../out/out_files/direct.out

# get pwms using only positive test cases (Y = 1)
python get_pwm.py --out_tag="max_active_per_seq" --parallel --include_y="observed" > ../out/out_files/max_active_per_seq_yobserved_1.out

# get pwms using only predicted strongly position cases (Yhat > .9)
python get_pwm.py --out_tag="max_active_per_seq" --parallel --include_y="predicted" > ../out/out_files/max_active_per_seq_ypredicted_0.9.out