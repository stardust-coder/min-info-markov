# Minimum information Markov model for EEG data analysis

Minimum information Markov model is suitable for EEG connectivity estimation for three reasons:
- temporal dependence
- any domain (phase, amplitude etc.)
- unified estimation framework

## Components

- Minimum information Markov model 
- GroupLASSO/Sparse estimation/Parallel computation
- FISTA 
- SGD (coming soon...) 
- Automatic model selection with information criteria (coming soon...) 
- Graphical analysis (coming soon...) 

## Example Usage

### Phase-phase connectivity
1. Set dependence functions.

`torus_pair_feature()` function defines a set of suitable dependences for phase-phase connectivity analysis.


2. Run estimation.

For 5 dimensional data
```
python run_PPC.py --dim 5 --order 1 --method pmle_grouplasso
```

### Phase-amplitude coupling

1. Set dependence functions.

2. Run estimation.

For 2 dimensional data
```
python run_PAC.py --dim 2 --order 1 --method pmle_fista
```


## Simulation dataset

`data.py` contains Kuramoto model with structured coefficient matrix $K$. 

## Real dataset

Coming soon...


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