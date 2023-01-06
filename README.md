# TaskRouter
First, you need to install this package:
```shell
pip install -e .
```

## Running CIFAR-MTL hyperparameter tuning
If you want to route all layers:
```shell
python3 main.py --dataset cifar --all_layers --hyperparameter_tune 
```

If you only want to route the FC layers
```shell
python3 main.py --dataset cifar --hyperparameter_tune 

```