#!/bin/zsh

eps_list=(0.06 0.05 0.04 0.03 0.02 0.01)
datasets=(covtype ijcnn)

for d in $datasets
do
    mkdir -p $d && cd $d
    mkdir -p runs output wandb_logs summary
    mv ../$d*.csv runs
    mv ../output/$d* output
    mv ../err_msgs/$d* wandb_logs

    for eps in $eps_list
        do awk '(NR == 1) || (FNR > 1)' runs/*=$eps*.csv > summary/${d}_eps_g=$eps.csv
    done
    cd ..
done