# !/bin/bash
SECONDS=0

for i in $(seq 1 100); do
  echo "=== Run ${i}/100 ==="

  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  python run_PPC2_gpu_cv.py \
    --x-npy X.npy \
    --build-x \
    --overwrite-x \
    --dim 5 \
    --order 1 \
    --gpus 0,1,2,3 \
    --cv-method kfold \
    --cv-folds 100 \
    --cv-shuffle \
    --cv-seed "$((12344 + i))" \
    --output-dir "./logs/cv/${i}"
done


# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# OMP_NUM_THREADS=1 \
# MKL_NUM_THREADS=1 \
# OPENBLAS_NUM_THREADS=1 \
# python run_PPC2_gpu_cv.py \
#   --x-npy X.npy \
#   --build-x \
#   --overwrite-x \
#   --dim 5 \
#   --order 1 \
#   --gpus 0,1,2,3 \
#   --cv-method loocv \
#   --output-dir ./logs/cv 
  

time=$SECONDS
echo $time