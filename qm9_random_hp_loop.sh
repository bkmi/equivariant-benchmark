#!/usr/bin/env bash
runs="$1"
folder="$2"
counter=0
while [ $counter -lt $runs ]
do
    python qm9_random_hp_search.py "$folder" qm9.db pst.npz
    counter=$(( $counter + 1 ))
done