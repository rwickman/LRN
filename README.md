# TaskRouter
## Running Routing Test
```shell
python3 main.py --train_data data/binary-mnist-mtl.npy --test_data data/binary-mnist-mtl-test.npy --epochs 1 --load_constraints contraints/no_shared.json --num_dup_layers 2 --max_depth 3
```

## Running CIFAR-MTL
```shell
python3 main.py --batch_size 2 --max_depth 4 --num_dup_layer 4 --epochs 10 --train_data data/cifar-mtl.npy --test_data data/cifar-mtl-test.npy --in_channels 3 --num_tasks 20 --img_size 32 --num_classes 5
```
