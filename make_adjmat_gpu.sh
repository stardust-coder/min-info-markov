# #!/usr/bin/env bash
python make_adjmat_gpu.py \
    --output-dir ./logs/marmoset20-2/999/theta \
    --dim 20 \
    --order 1 \
    --ground-truth ./25dim \
    --threshold 1e-8

# for number in $(seq 926 992); do
#     output_dir="./logs/marmoset20-2/${number}/theta"

#     if [[ ! -d "$output_dir" ]]; then
#         echo "スキップ: ディレクトリがありません: $output_dir" >&2
#         continue
#     fi

#     echo "実行中: $number"

#     python make_adjmat_gpu.py \
#         --output-dir "$output_dir" \
#         --dim 20 \
#         --order 1 \
#         --ground-truth ./25dim \
#         --threshold 1e-8
# done

echo "すべて完了しました。"