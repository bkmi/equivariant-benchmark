#!/bin/bash
today=$(date +%Y%m%d)
db="qm9.db"

case "$1" in
  small)
    # Small
    targets="A B C mu alpha homo lumo gap r2 zpve U0 U H G Cv"
    split_file="default_split.npz"

    for target in $targets
    do
      model_dir=${today}_${target}
      python qm9_train.py --model_dir "$model_dir" --split_file "$split_file" --db "$db" --wall 3500 --"$target"
    done
    ;;

  paper)
    # Paper sized
    target="U0"
    split_file="paper_split.npz"
    model_dir=${2:-${today}_${target}_paper}
    python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 43200 \
        --"$target" \
        --ntr 109000 \
        --nva 1000 \
        --bs 16
    ;;

  paper_l1)
    # Paper sized with l1
    target="U0"
    split_file="paper_split.npz"
    model_dir=${2:-${today}_${target}_paper_l1}
    python qm9_train.py \
      --model_dir "$model_dir" \
      --split_file "$split_file" \
      --db "$db" \
      --wall 43200 \
      --"$target" \
      --ntr 109000 \
      --nva 1000 \
      --bs 16 \
      --l0 32 \
      --l1 10
    ;;

  *)
    echo "$1" is not a possible argument.
    exit 1
esac