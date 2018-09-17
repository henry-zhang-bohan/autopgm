# autopgm
Automatically train a merged Bayesian Network from multiple data sources

# Installation
Using pip: open `autopgm` folder, and run
```unix
$ pip3 install .
```

# Bayesian Data Integration

## Model training
In `python3`, run
```python
from autopgm.estimator import MultipleBayesianEstimator
model = MultipleBayesianEstimator([csv_file_name_1, csv_file_name_2, ...]).get_model()
```
Note: all files need to be .csv files with discrete variables of integer values.

## Inference
`model` is a `BayesianModel` as in `pgmpy`.
You can perform `VariableElimination` and then `query` as in `pgmpy`:
```python
from pgmpy.inference import VariableElimination
inference = VariableElimination(model)
q = inference.query(['var1'], evidence={'var2': 0, 'var3': 1})['var1']
```
