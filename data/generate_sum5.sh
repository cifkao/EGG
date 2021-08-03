#!/bin/bash
set -ex
python ../egg/zoo/basic_games/data_generation_scripts/generate_summation_dataset.py --prefix sum5. --test-prob 0.25 --resample 5 50
shuf sum5.train >sum5.train.shuf
head -n -50 sum5.train.shuf >sum5.train.train
tail -n 50 sum5.train.shuf >sum5.train.val
wc -l sum5.*
