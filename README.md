# Minimum information Markov model for EEG data analysis

See the latest repo [coming soon]().


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

## Other files

- sampling.py: sample from predefined easy model for bias correction experiments.
- bias_check.py: bias correction experiments.
- bias_post.py: same. run after above.
- fista.py: implements FISTA optimization algorithm used in lasso.
- sinkhorn.py: for comparing various circular time series models.
- besag.py: not used.