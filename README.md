# Minimum information Markov modeling

This branch implements the pseudo likelihood estimation on the minimum information Markov model for time series data.

## Basic Usage
1. Set data. (l96~98)
    - Use simulation data from AR processes if needed.
2. Set model parameters. (l99~100)
3. Choose estimation method (l230~245)
    - For AR models, you may compare with MLE.
    - We recommend PLE (default).

4. Run estimation.
```
python run.py
```

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
```
Coming soon...
```