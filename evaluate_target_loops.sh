#!/bin/bash
prefix="$1"
for directory in "$prefix"/*/
do
  python qm9_train.py --model_dir "$directory" --db qm9.db --wall 0
done