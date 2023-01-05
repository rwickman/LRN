import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np

sns.set(style="darkgrid")

train_dict_file = "train_models/cifar_only_fc/models_balcvloss01_mintemp0001_tempdecay58500_normfirst_warmup1500_dup20_lr1e4_routerlr1e4_stable_tanh/train_dict.json"

with open(train_dict_file) as f:
    train_dict = json.load(f)

def moving_average(x, w=3):
    return np.convolve(x, np.ones(w), 'valid') / w

fig, axs = plt.subplots(4)


axs[0].plot(moving_average(train_dict["train_loss"]))
axs[0].set(xlabel="Epoch", ylabel="Training Loss")
axs[1].plot(moving_average(train_dict["val_loss"]))
axs[1].set(xlabel="Epoch", ylabel="Validation Loss")
axs[2].plot(moving_average(train_dict["val_acc"]))
axs[2].set(xlabel="Epoch", ylabel="Validation Accuracy")
if "bal_cv_loss" in train_dict:
    axs[3].plot(moving_average(train_dict["bal_cv_loss"]))
    axs[3].set(xlabel="Epoch", ylabel="Bal CV Loss")

print("NUM EPOCHS", len(train_dict["train_loss"]))
print("\nVAL ACC", train_dict["val_acc"][-10:])
print("\nTrain Loss", train_dict["train_loss"][-10:])
if "bal_cv_loss" in train_dict:
    print("\nBAL CV LOSS", train_dict["bal_cv_loss"][-10:])

# print("\ntrain loss", train_dict["train_loss"
# ])
# print("val loss", train_dict["val_loss"])

# print(train_dict["train_modules"])
# print(train_dict["val_modules"])

# print("\nTRAIN: ", train_dict["train_modules"][-1])
# print("VAL: ", train_dict["val_modules"][-1])

print("\nTRAIN LAYERS")
num_modules = 20
router_depth = 3
for i in range(router_depth):
    print(train_dict["train_modules"][-1][i * num_modules: (i+1)*num_modules])

print("\n\nVAL LAYERS")
for i in range(router_depth):
    print(train_dict["val_modules"][-1][i * num_modules: (i+1)*num_modules])


# print("\nTRAIN INPUT")
# print(train_dict["train_modules"][-1][:3])

# print("\nTRAIN HIDDEN")
# for i in range(4):
#     print(train_dict["train_modules"][-1][3 + 6*i:3 + 6*(i+1)])
# print("\nTRAIN DOWNSAMPLE")
# for i in range(4):
#     print(train_dict["train_modules"][-1][3 + 6*6 + i*3: 3 + 6*6 + (i+1)*3])
# # print("\nTRAIN OUTPUT")
# # print(train_dict["train_modules"][-1][-20:])



# print("\n\nVAL INPUT")
# print(train_dict["val_modules"][-1][:3])

# print("\nVAL HIDDEN")
# for i in range(4):
#     print(train_dict["val_modules"][-1][3 + 6*i:3 + 6*(i+1)])
# print("\nVAL DOWNSAMPLE")
# for i in range(4):
#     print(train_dict["val_modules"][-1][3 + 6*6 + i*3: 3 + 6*6 + (i+1)*3])
# print("\nVAL OUTPUT")
# print(train_dict["val_modules"][-1][-20:])
plt.show()
