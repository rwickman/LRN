#!/bin/bash

alpha_vals=(0.01 0.05 0.1)
load_bal_vals=(0.01 0.1 1.0)
fb_drop_vals=(0.1 0.2 0.3 0.4 0.5)
epochs=50
max_depth=4
num_dup_layer=2


# Nothing
python3 main.py --batch_size 64 --max_depth $max_depth --num_dup_layer $num_dup_layer --epochs $epochs \
  --train_data data/cifar_reptile_people.npy --test_data data/cifar_reptile_people_test.npy \
  --in_channels 3 --num_task 2 --img_size 32 --num_classes 5 --critic_lam 0.1 \
  --fb_drop_rate 0.0 --load_bal_lam 0.0 --alpha 0 --smooth_lam 0 --no_reward_bonus
mv models "old_models/CIFAR_Reptile_People/val/models_load_0_drop_0_alpha_0_nobonus"

# Reward bonus only
python3 main.py --batch_size 64 --max_depth $max_depth --num_dup_layer $num_dup_layer --epochs $epochs \
  --train_data data/cifar_reptile_people.npy --test_data data/cifar_reptile_people_test.npy \
  --in_channels 3 --num_task 2 --img_size 32 --num_classes 5 --critic_lam 0.1 \
  --fb_drop_rate 0.0 --load_bal_lam 0.0 --alpha 0 --smooth_lam 0
mv models "old_models/CIFAR_Reptile_People/val/models_load_0_drop_0_alpha_0"

# Label smoothing only
for v in "${alpha_vals[@]}";
do
  python3 main.py --batch_size 64 --max_depth $max_depth --num_dup_layer $num_dup_layer --epochs $epochs \
    --train_data data/cifar_reptile_people.npy --test_data data/cifar_reptile_people_test.npy \
    --in_channels 3 --num_task 2 --img_size 32 --num_classes 5 --critic_lam 0.1 \
    --fb_drop_rate 0.0 --load_bal_lam 0.0 --alpha $v --no_reward_bonus 
  mv models "old_models/CIFAR_Reptile_People/val/models_load_0_drop_0_alpha_${v}_nobonus"
done

# Load balance only
for v in "${load_bal_vals[@]}";
do
  python3 main.py --batch_size 64 --max_depth $max_depth --num_dup_layer $num_dup_layer --epochs $epochs \
    --train_data data/cifar_reptile_people.npy --test_data data/cifar_reptile_people_test.npy \
    --in_channels 3 --num_task 2 --img_size 32 --num_classes 5 --critic_lam 0.1 \
    --fb_drop_rate 0.0 --load_bal_lam $v --alpha 0 --smooth_lam 0 --no_reward_bonus 
  mv models "old_models/CIFAR_Reptile_People/val/models_load_${v}_drop_0_alpha_0_nobonus"
done


# FB Dropout only
for v in "${fb_drop_vals[@]}";
do
  python3 main.py --batch_size 64 --max_depth $max_depth --num_dup_layer $num_dup_layer --epochs $epochs \
    --train_data data/cifar_reptile_people.npy --test_data data/cifar_reptile_people_test.npy \
    --in_channels 3 --num_task 2 --img_size 32 --num_classes 5 --critic_lam 0.1 \
    --fb_drop_rate $v --load_bal_lam 0 --alpha 0 --smooth_lam 0 --no_reward_bonus 
  mv models "old_models/CIFAR_Reptile_People/val/models_load_0_drop_${v}_alpha_0_nobonus"
done
