#!/bin/bash
today=$(date +%Y%m%d)
db="qm9.db"
prefix="$2"

case "$1" in
  small)
    # Small
    targets="A B C mu alpha homo lumo gap r2 zpve U0 U H G Cv"
    split_file="default_split.npz"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}
      python qm9_train.py --model_dir "$model_dir" --split_file "$split_file" --db "$db" --wall 3500 --"$target"
    done
    ;;

  big)
    targets="A B C mu alpha homo lumo gap r2 zpve U0 U H G Cv"
    split_file="paper_split.npz"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}_64
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 43200 \
        --"$target" \
        --ntr 109000 \
        --nva 1000 \
        --bs 16
    done
    ;;

  big_narrow)
    targets="A B C mu alpha homo lumo gap r2 zpve U0 U H G Cv"
    split_file="paper_split.npz"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}_64
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 43200 \
        --"$target" \
        --ntr 109000 \
        --nva 1000 \
        --bs 16 \
        --l0 32
    done
    ;;

  big3)
    targets="A B C mu alpha homo lumo gap r2 zpve U0 U H G Cv"
    split_file="paper_split.npz"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}_64
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 43200 \
        --"$target" \
        --ntr 109000 \
        --nva 1000 \
        --bs 16 \
        --L 3
    done
    ;;

  big_l1)
    targets="A B C mu alpha homo lumo gap r2 zpve U0 U H G Cv"
    split_file="paper_split.npz"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}_64_l1
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
    done
    ;;

  big_narrow_l1)
    targets="A B C mu alpha homo lumo gap r2 zpve U0 U H G Cv"
    split_file="paper_split.npz"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}_64_l1
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 43200 \
        --"$target" \
        --ntr 109000 \
        --nva 1000 \
        --bs 16 \
        --l0 16 \
        --l1 5
    done
    ;;

  big3_l1)
    targets="A B C mu alpha homo lumo gap r2 zpve U0 U H G Cv"
    split_file="paper_split.npz"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}_64_l1
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
        --l1 10 \
        --L 3
    done
    ;;

  u0_64)
    # Half schnet paper sized
    target="U0"
    split_file="paper_split.npz"
    model_dir=${2:-${today}_${target}_u0_64}
    python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 86400 \
        --"$target" \
        --ntr 109000 \
        --nva 1000 \
        --bs 16
    ;;

  u0_64_res)
    # Half schnet paper sized with resnet
    target="U0"
    split_file="paper_split.npz"
    model_dir=${2:-${today}_${target}_u0_64_res}
    python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 86400 \
        --"$target" \
        --ntr 109000 \
        --nva 1000 \
        --bs 16 \
        --res
    ;;

  u0_64_l1_res)
    # Half schnet paper sized with resnet and l1
    target="U0"
    split_file="paper_split.npz"
    model_dir=${2:-${today}_${target}_u0_64_l1_res}
    python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 86400 \
        --"$target" \
        --ntr 109000 \
        --nva 1000 \
        --bs 16 \
        --l0 32 \
        --l1 10
    ;;

  u0_64_l1)
    # Half schnet paper sized with l1
    target="U0"
    split_file="paper_split.npz"
    model_dir=${2:-${today}_${target}_u0_64_l1}
    python qm9_train.py \
      --model_dir "$model_dir" \
      --split_file "$split_file" \
      --db "$db" \
      --wall 86400 \
      --"$target" \
      --ntr 109000 \
      --nva 1000 \
      --bs 16 \
      --l0 32 \
      --l1 10
    ;;

  u0_128)
    # Paper sized
    target="U0"
    split_file="paper_split.npz"
    model_dir=${2:-${today}_${target}_u0_128}
    python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 86400 \
        --"$target" \
        --ntr 109000 \
        --nva 1000 \
        --bs 20 \
        --l0 128 \
        --embed 128 \
        --rad_h 128
    ;;

  # Added 20.02.2020
  mu_u0_128_res)
    targets="mu U0"
    split_file="paper_split_tesseract.npz"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 86400 \
        --"$target" \
        --ntr 109000 \
        --nva 1000 \
        --bs 20 \
        --l0 128 \
        --embed 128 \
        --rad_h 128 \
        --res \
        --reduce_lr_patience 6
    done
    ;;

  # Added 20.02.2020
  mu_u0_128_l1_res)
    targets="mu U0"
    split_file="paper_split_tesseract.npz"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 86400 \
        --"$target" \
        --ntr 109000 \
        --nva 1000 \
        --bs 20 \
        --l0 64 \
        --l1 21 \
        --embed 128 \
        --rad_h 128 \
        --res \
        --reduce_lr_patience 6
    done
    ;;

  # Added 20.02.2020
  mu_u0_128_l1l2_res)
    targets="mu U0"
    split_file="paper_split_tesseract.npz"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 86400 \
        --"$target" \
        --ntr 109000 \
        --nva 1000 \
        --bs 20 \
        --l0 42 \
        --l1 14 \
        --l2 8 \
        --embed 128 \
        --rad_h 128 \
        --res \
        --reduce_lr_patience 6
    done
    ;;

  # Added 28.02.2020
  mu_u0_128_res_gauss)
    targets="mu U0"
    split_file="paper_split_tesseract.npz"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 86400 \
        --"$target" \
        --ntr 109000 \
        --nva 1000 \
        --bs 20 \
        --l0 128 \
        --embed 128 \
        --rad_h 128 \
        --res \
        --reduce_lr_patience 6 \
        --radial_model gaussian
    done
    ;;

  # Added 28.02.2020
  mu_u0_128_l1_res_gauss)
    targets="mu U0"
    split_file="paper_split_tesseract.npz"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 86400 \
        --"$target" \
        --ntr 109000 \
        --nva 1000 \
        --bs 20 \
        --l0 64 \
        --l1 21 \
        --embed 128 \
        --rad_h 128 \
        --res \
        --reduce_lr_patience 6 \
        --radial_model gaussian
    done
    ;;

  # Added 22.02.2020
  continue)
    for i in "${@:2}" # iterate through second argument and beyond
    do
      model_dir="$i"
      python qm9_train.py \
        --model_dir "$model_dir" \
        --wall 86400 \
        --split_file this_is_not_used \
        --db this_is_not_used
    done
    ;;

  *)
    echo "$1" is not a possible argument.
    exit 1
esac
