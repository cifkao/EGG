#!/bin/bash
set -ex
python ../egg/zoo/basic_games/data_generation_scripts/generate_summation_dataset.py --prefix sum50. --test-prob 0.25 --resample 50 500
shuf sum50.train >sum50.train.shuf
head -n -500 sum50.train.shuf >sum50.train.train
tail -n 500 sum50.train.shuf >sum50.train.val
wc -l sum50.*
