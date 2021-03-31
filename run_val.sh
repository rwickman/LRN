#!/bin/bash

vals=(0.0 0.01 0.1 1.0)
epochs=25
for v in "${vals[@]}";
do
  python3 main.py --batch_size 64 --max_depth 6 --num_dup_layer 5 --epochs $epochs --train_data data/cifar-mtl.npy \
	  --test_data data/cifar-mtl-test.npy --in_channels 3 --num_task 20 --img_size 32 --num_classes 5 \
	  --fb_drop_rate 0.0 --critic_lam 0.1 --load_bal_lam $v --no_reward_bonus --val_split
  mv models "old_models/CIFAR/val/models_load_${v}_drop_0_nobonus"
done

vals=(0.1 0.2 0.3 0.4 0.5)
for v in "${vals[@]}";
do
  python3 main.py --batch_size 64 --max_depth 6 --num_dup_layer 5 --epochs $epochs --train_data data/cifar-mtl.npy \
	  --test_data data/cifar-mtl-test.npy --in_channels 3 --num_task 20 --img_size 32 --num_classes 5 \
	  --fb_drop_rate $v --critic_lam 0.1 --load_bal_lam 0.0 --no_reward_bonus --val_split
  mv models "old_models/CIFAR/val/models_load_0_drop_${v}_nobonus"
done
