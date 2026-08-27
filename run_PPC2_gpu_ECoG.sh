# !/bin/bash
SECONDS=0


for i in $(seq 120 198); do
    echo "=== Run ${i}/1000 ==="

    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    python run_PPC2_gpu.py \
        --x-npy ./X.npy \
        --build-x \
        --dim 20 \
        --order 1 \
        --num-lambdas 100 \
        --lambda-scale-by-nrows \
        --gpus 0,1,2,3 \
        --compute-ic \
        --fista-ridge 1e-6 \
        --refit-ridge 5e-7 \
        --nw-bandwidth -1 \
        --ic-chunk-rows 50000 \
        --output-dir "./logs/marmoset_Rc/${i}/gamma" \
        --dtype float32 \
        --support-abs-tol 1e-8 \
        --overwrite-x \
        --stim-index ${i}
done
time=$SECONDS
echo $time