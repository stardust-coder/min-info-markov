# Minimum information Markov model for EEG data analysis


## Usage

1. run simulation experiments.

```
pip install -r requirements.txt

# check data.py, where ground truth Kuramoto model is implemented, then 
python run_PPC.py --dim 25 --order 1 --method pmle_grouplasso --save_dir "logs"
# or 
bash run.sh
```

The numpy weights will be saved at `--save_dir`.

2. get results.

```
#Adjust DIR_NAME at the beginning. Then, 
python make_adjmat.py
```

Plot images of adjancy matrix, Plot of information criteria vs num of edges, csv will be stored in the same `--save_dir` directory.

## Requirements

Our experiments were conducted on 4 NVIDIA A100 (40GB) GPU environments.

The peak memory usage was about 15GB each.


## Script for experiments

⚠️Warning. The current code saves very large `X.npy` and `X.npy.partial` (temporary) to the current directory. Be careful to the size of your storage.

- run_PPC2_gpu.py
    - main experiment codes.
    - need to select dataset manually by commenting out.
- run_PPC2_gpu_ECoG.sh
    - analyse marmoset auditory ECoG dataset by [Chao et al.](https://dataportal.brainminds.jp/ecog-auditory-02)
- run_PPC2_gpu_EEG.sh
    - analyse marmoset auditory ECoG dataset by [Chennu et al.](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004669)
- run_PPC2_gpu_single.sh
    - single run of run_PPC2_gpu.py

## Dataloaders

- data_sim.py: contain Kuramoto model and other simulations
- data_real.py: read ECoG dataset and EEG dataset. include metadata.


## Other files

- feature.py: build torus features for PLE. It corresponds to the design matrix of a large-scale logistic regression.
- feature_rotational.py: when you want to switch between multiplicative features for the full model and the rotational model with less parameters.
- sampling.py: sample from predefined easy model for bias correction experiments.
- bias_check.py: bias correction experiments.
- bias_post.py: same. run after above.
- fista.py: implements FISTA optimization algorithm used in lasso.
- sinkhorn.py: for comparing various circular time series models.
- besag.py: not used anymore.
- AR-TG.py: for comparison against baselines. reference: [Goffinet et al., ICML2026](https://arxiv.org/abs/2606.00496)