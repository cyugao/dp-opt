#!/bin/zsh
# Run the experiment with different parameters eps_g_target
for eps in 0.04 0.03 0.02
do
    echo Machine $fg[green]$: $reset_color$fg[red]`uptime | sed -r 's/.*: ([0-9]+\.[0-9]+).*/\1/'` $reset_color;
    python3 main.py $eps > "output/eps_g_target_$eps.txt"
    # awk 'FNR==1 && NR!=1{next;}{print}' *.csv

    # echo "eps_g_target = $eps"
done