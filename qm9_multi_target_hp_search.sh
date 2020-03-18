#!/usr/bin/env bash
today=$(date +%Y%m%d)
db="qm9.db"
split_file="pst.npz"

case "$1" in
    gleichzeitig)
        model_dir=${2:-${today}_gleichzeitig}
        python qm9_train.py \
            --model_dir "$model_dir" \
            --split_file "$split_file" \
            --db "$db"  \
            --wall 87120 \
            --L 5 \
            --l0 128 \
            --embed 128 \
            --rad_h 128 \
            --res \
            --mu \
            --alpha \
            --homo \
            --lumo \
            --gap \
            --r2 \
            --zpve \
            --U0 \
            --U \
            --H \
            --G \
            --Cv
        ;;

    gleichzeitig_l1)
        model_dir=${2:-${today}_gleichzeitig_l1}
        python qm9_train.py \
            --model_dir "$model_dir" \
            --split_file "$split_file" \
            --db "$db"  \
            --wall 87120 \
            --L 5 \
            --l0 64 \
            --l1 21 \
            --embed 128 \
            --rad_h 128 \
            --res \
            --mu \
            --alpha \
            --homo \
            --lumo \
            --gap \
            --r2 \
            --zpve \
            --U0 \
            --U \
            --H \
            --G \
            --Cv
        ;;

    gleichzeitig_bessel_mlp)
        model_dir=${2:-${today}_gleichzeitig_bessel_mlp}
        python qm9_train.py \
            --model_dir "$model_dir" \
            --split_file "$split_file" \
            --db "$db"  \
            --wall 87120 \
            --l0 128 \
            --embed 128 \
            --rad_h 128 \
            --radial_model bessel \
            --mlp_out \
            --res \
            --mu \
            --alpha \
            --homo \
            --lumo \
            --gap \
            --r2 \
            --zpve \
            --U0 \
            --U \
            --H \
            --G \
            --Cv
        ;;

    gleichzeitig_l1_bessel_mlp)
        model_dir=${2:-${today}_gleichzeitig_l1_bessel_mlp}
        python qm9_train.py \
            --model_dir "$model_dir" \
            --split_file "$split_file" \
            --db "$db"  \
            --bs 14 \
            --wall 87120 \
            --l0 48 \
            --l1 16 \
            --embed 96 \
            --rad_h 96 \
            --radial_model bessel \
            --mlp_out \
            --outnet_l0 48 \
            --outnet_l1 16 \
            --res \
            --mu \
            --alpha \
            --homo \
            --lumo \
            --gap \
            --r2 \
            --zpve \
            --U0 \
            --U \
            --H \
            --G \
            --Cv
        ;;

    *)
        echo "$1" is not a possible argument.
        exit 1
esac