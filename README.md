# CardiacCTAnalysis
Master Thesis Project for building machine learning models to analyze CT scans of hearts.


## Examples
Runnning the following will give you the guide to using the tool.
```
python3 main.py -h
```

A training run could look like this
```
python3 main.py -t train -i 2 --samples 5 -s 750 -e 50
```
With a new image being loaded every 2 episodes the bezier curve being sampled 5 different places, each episode running for max 750 steps and 50 episodes being run.

And an evaluation could look like
```
python3 main.py -t eval --samples 5 -s 100 -e 10
```


