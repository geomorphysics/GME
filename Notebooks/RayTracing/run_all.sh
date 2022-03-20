#!/usr/bin/env bash

# Master wrapper script to execute all notebook jobs

for nb_group in TimeInvariant OneRay VelocityBoundary 
do
    printf "\nRunning nb group: $nb_group\n"
    ../run_jobs.sh $nb_group
    echo
done
