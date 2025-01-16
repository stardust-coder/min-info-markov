# Minimum information Markov model


# Usage : simulating VAR model

`VAR.py` is for when VAR parameters A are the target and the error covariance Σ is known. 

1. Write a config like the following sample and place like `config/var_config.json`.
    <details>
        <summary>config sample</summary>
    {
        "model": "VAR",
        "order": 1,
        "dim": 2,
        "phi": [
            [
                [
                    0.5,
                    0.1
                ],
                [
                    0.1,
                    0.5
                ]
            ]
        ],
        "sigma": [
            [
                0.5,
                0
            ],
            [
                0,
                0.5
            ]
        ],
        "mean": [
            0,
            0
        ],
        "steps": 1000,
        "run": 30
    }
    </details>

2. Specify a config file to run an experiment.
```
python VAR.py config/hogehoge.json
```







# How to cite
```
Coming soon...
```
