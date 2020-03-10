#!/bin/bash
today=$(date +%Y%m%d)
db="qm9.db"
prefix="$2"
split_file="pst.npz"

case "$1" in
  small)
    # Small
    targets="mu alpha homo lumo gap r2 zpve U0 U H G Cv"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}
      python qm9_train.py --model_dir "$model_dir" --split_file "$split_file" --db "$db" --wall 3500 --"$target"
    done
    ;;

  big)
    targets="mu alpha homo lumo gap r2 zpve U0 U H G Cv"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}_64
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 43200 \
        --"$target"
    done
    ;;

  big_narrow)
    targets="mu alpha homo lumo gap r2 zpve U0 U H G Cv"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}_64
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 43200 \
        --"$target" \
        --l0 32
    done
    ;;

  big3)
    targets="mu alpha homo lumo gap r2 zpve U0 U H G Cv"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}_64
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 43200 \
        --"$target" \
        --L 3
    done
    ;;

  big_l1)
    targets="mu alpha homo lumo gap r2 zpve U0 U H G Cv"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}_64_l1
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 43200 \
        --"$target" \
        --l0 32 \
        --l1 10
    done
    ;;

  big_narrow_l1)
    targets="mu alpha homo lumo gap r2 zpve U0 U H G Cv"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}_64_l1
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 43200 \
        --"$target" \
        --l0 16 \
        --l1 5
    done
    ;;

  big3_l1)
    targets="mu alpha homo lumo gap r2 zpve U0 U H G Cv"
    
    for target in $targets
    do
      model_dir=${prefix}${today}_${target}_64_l1
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db" \
        --wall 43200 \
        --"$target" \
        --l0 32 \
        --l1 10 \
        --L 3
    done
    ;;

  u0_64)
    # Half schnet paper sized
    target="U0"
    
    model_dir=${2:-${today}_${target}_u0_64}
    python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db"  \
        --"$target" \
        --ntr 109000 \
        --nva 1000 \
        --bs 16
    ;;

  u0_64_res)
    # Half schnet paper sized with resnet
    target="U0"
    
    model_dir=${2:-${today}_${target}_u0_64_res}
    python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db"  \
        --"$target" \
        --res
    ;;

  u0_64_l1_res)
    # Half schnet paper sized with resnet and l1
    target="U0"
    
    model_dir=${2:-${today}_${target}_u0_64_l1_res}
    python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db"  \
        --"$target" \
        --l0 32 \
        --l1 10
    ;;

  u0_64_l1)
    # Half schnet paper sized with l1
    target="U0"
    
    model_dir=${2:-${today}_${target}_u0_64_l1}
    python qm9_train.py \
      --model_dir "$model_dir" \
      --split_file "$split_file" \
      --db "$db"  \
      --"$target" \
      --l0 32 \
      --l1 10
    ;;

  u0_128)
    # Paper sized
    target="U0"
    
    model_dir=${2:-${today}_${target}_u0_128}
    python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db"  \
        --"$target" \
        --l0 128 \
        --embed 128 \
        --rad_h 128
    ;;

  # Added 20.02.2020
  mu_u0_128_res)
    targets="mu U0"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db"  \
        --"$target" \
        --l0 128 \
        --embed 128 \
        --rad_h 128 \
        --res
    done
    ;;

  # Added 20.02.2020
  mu_u0_128_l1_res)
    targets="mu U0"
    
    for target in $targets
    do
      model_dir=${prefix}${today}_${target}
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db"  \
        --"$target" \
        --l0 64 \
        --l1 21 \
        --embed 128 \
        --rad_h 128 \
        --res
    done
    ;;

  # Added 20.02.2020
  mu_u0_128_l1l2_res)
    targets="mu U0"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db"  \
        --"$target" \
        --l0 42 \
        --l1 14 \
        --l2 8 \
        --embed 128 \
        --rad_h 128 \
        --res
    done
    ;;

  # Added 28.02.2020
  mu_u0_128_res_gauss)
    targets="mu U0"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db"  \
        --"$target" \
        --l0 128 \
        --embed 128 \
        --rad_h 128 \
        --res \
        --radial_model gaussian
    done
    ;;

  # Added 28.02.2020
  mu_u0_128_l1_res_gauss)
    targets="mu U0"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db"  \
        --"$target" \
        --l0 64 \
        --l1 21 \
        --embed 128 \
        --rad_h 128 \
        --res \
        --radial_model gaussian
    done
    ;;

  # Added 09.03.2020
  all_l1_no_mu)
    targets="alpha homo lumo gap r2 zpve U0 U H G Cv"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db"  \
        --"$target" \
        --l0 64 \
        --l1 21 \
        --embed 128 \
        --rad_h 128 \
        --res \
        --wall 87120
    done
    ;;

  # Added 09.03.2020
  all_no_mu_u0)
    targets="alpha homo lumo gap r2 zpve U H G Cv"

    for target in $targets
    do
      model_dir=${prefix}${today}_${target}
      python qm9_train.py \
        --model_dir "$model_dir" \
        --split_file "$split_file" \
        --db "$db"  \
        --"$target" \
        --l0 128 \
        --embed 128 \
        --rad_h 128 \
        --res \
        --wall 87120
    done
    ;;

  # Added 22.02.2020
  continue)
    for i in "${@:2}" # iterate through second argument and beyond
    do
      model_dir="$i"
      python qm9_train.py \
        --model_dir "$model_dir"  \
        --split_file this_is_not_used \
        --db this_is_not_used
    done
    ;;

  *)
    echo "$1" is not a possible argument.
    exit 1
esac
