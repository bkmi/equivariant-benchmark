#!/usr/bin/env bash
RUNS="$1"
FOLDER="$2"
COUNTER=0
while [$COUNTER -lt RUNS]; do
    python qm9_random_hp_search.py "$FOLDER" qm9.db pst.npz
    COUNTER=$(( $COUNTER + 1 ))
done