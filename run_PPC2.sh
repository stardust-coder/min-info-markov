# !/bin/bash
SECONDS=0

OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    python run_PPC2.py --dim 20 --order 1 --method pmle_grouplasso --save_dir "logs/marmoset20-2/2/beta"

time=$SECONDS
echo $time