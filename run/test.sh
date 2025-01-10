#! /bin/bash
test_data_dir="SfPUEL_test_data/real1"
save_folder="sfpuel_real1"
python tools/test.py --data_dir_tst $test_data_dir --suffix $save_folder &&\
    python tools/normal_criterion.py -d results/$save_folder --gt_data_dir $test_data_dir 
