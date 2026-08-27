# !/bin/bash
SECONDS=0


list=(2 3 5 6 7 8 9 10 13 14 18 20 22 23 24 25 26 27 28 29)

for i in "${list[@]}"; do
    OUTDIR="./logs/human/${i}/alpha/baseline"
    
    echo "=== Run Patient ${i} ==="

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    python run_PPC2_gpu.py \
        --x-npy ./X.npy \
        --build-x \
        --dim 19 \
        --order 1 \
        --num-lambdas 100 \
        --lambda-scale-by-nrows \
        --gpus 0,1,2,3 \
        --compute-ic \
        --fista-ridge 5e-5 \
        --refit-ridge 5e-5 \
        --nw-bandwidth -1 \
        --ic-chunk-rows 50000 \
        --output-dir "$OUTDIR" \
        --dtype float32 \
        --support-abs-tol 1e-8 \
        --overwrite-x \
        --stim-index ${i}
done

time=$SECONDS
echo $time
