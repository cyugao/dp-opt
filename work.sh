#!/bin/bash
# time source /mnt/home/cgao/miniforge3/etc/profile.d/conda.sh
# time conda activate ml
source /mnt/home/cgao/miniforge3/bin/activate ml
python main.py "$@"
# time python3 main.py 0.06 "dptr" >> "output/eps_g_0.06.txt"