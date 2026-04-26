# Minimum information Markov model

This branch implements the minimum information Markov model for time series data.

## Basic Usage
1. Set data. (l96~98)
    - Use simulation data from AR processes if needed.
2. Set model parameters. (l99~100)
3. Choose estimation method (l255~271)
    - For AR models, you may compare with MLE.
    - We recommend PLE (default).
4. Set func_h used in PLE.
    - l175 defines the func_h `func_h = func_h_matrix`. Note that `func_h_naive` is very slow. Use equivalent accelerations `func_h_einsum` or `func_h_matrix(recommended)`.
5. Run estimation.
```
python run.py --dim 1 --order 1 --method pmle_sgd
python run.py --dim 56 --order 1 --method pmle_fista  #for sparse estimation
```

Note: The optimization procedure in PLE is in fact a standard logistic regression. The optimization method is a simple gradient descent with zero initialization and inverse time decayed learning rate.


## Simple example on univariate binary spike train data

This code is currently used only for plotting the data.
```
python run_spike.py
```


## Simple example on univariate LFP data

For each of ch1 ~ ch96, this code performs PLE with the dependence function
$$h(x,y) = (xy, x^2y, xy^2)^\top,$$
which is defined as "func_h_v2" in the code.
```
python run_LFP.py
```

## Cross domain analysis of LFP and spike trains

1. (l247) Set prepare = True for your initial run. After you have parallel.csv, set prepare = False.
2. (l257) Set unit (= electrode channel).
3. (l76~l93) Determine the dependence function "func_h", which will be defined by "func_h_custom" and $K$. 
4. Run estimation.
```
python run_LFP_Spike.py
```

## How to cite

Please consider citing the following preprint.

```
@article{sukeda2026minimum,
  title={Minimum information Markov model},
  author={Sukeda, Issey and Sei, Tomonari},
  journal={arXiv preprint arXiv:2601.06900},
  year={2026}
}
```