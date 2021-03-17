import torch
import torch.nn as nn
import torch.optim as optim
import os, sys
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

sys.path.append(os.path.abspath("../"))
from data_loader import DataLoader



bce_loss_fn = nn.BCEWithLogitsLoss()

def create_unshared(args):
    return nn.Sequential(
        nn.Conv2d(args.in_channels, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(args.img_size**2*32, 1)
    ).cuda()

def create_shared(args):
    shared_layer = nn.Conv2d(args.in_channels, 32, 3, stride=1, padding=1) 
    
    one_model = nn.Sequential(
        shared_layer,
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(args.img_size**2*32, 1)
    ).cuda()
    
    zero_model = nn.Sequential(
        shared_layer,
        nn.ReLU(),
        nn.Conv2d(32, 32, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(args.img_size**2*32, 1)
    ).cuda()
    
    return zero_model, one_model
    

def run_mnist_benchmarks(args):
    # zero_model = create_unshared(args)
    # one_model = create_unshared(args)
    zero_model, one_model = create_shared(args)
    
    
    zero_train_data, zero_test_data, one_train_data, one_test_data = create_mnist_data(args)    
    
    zero_model = train_model(args, zero_train_data, zero_model)
    one_model = train_model(args, one_train_data, one_model)

    zero_y_preds = test_model(args, zero_test_data, zero_model)
    one_y_preds = test_model(args, one_test_data, one_model)
    y_preds = np.vstack((zero_y_preds, one_y_preds))
    
    y_trues = np.vstack((zero_test_data[1], one_test_data[1]))
    
    test_acc_score = np.sum(y_trues == y_preds) / len(y_trues)
    print("Accuracy of testing dataset", test_acc_score)
    test_f1_score = f1_score(y_trues, y_preds)
    print("test_f1_score", test_f1_score)
    


    

def create_mnist_data(args):
    loader = DataLoader(args)
    train_data, test_data = loader.load_mnist()    
    
    def split_data(data):
        zero_imgs, zero_targets = [], []
        one_imgs, one_targets = [], []

        for ex in data:
            if ex[0] == 0:
                zero_imgs.append(ex[1][0])
                zero_targets.append([ex[2]])
            elif ex[0] == 1:
                one_imgs.append(ex[1][0])
                one_targets.append([ex[2]])
            else:
                assert False

        zero_data = [np.array(zero_imgs), np.array(zero_targets)]
        one_data = [np.array(one_imgs), np.array(one_targets)]
        return zero_data, one_data

    zero_train_data, one_train_data = split_data(train_data)
    zero_test_data, one_test_data = split_data(test_data)
    
    return zero_train_data, zero_test_data, one_train_data, one_test_data


def train_model(args, train_data, model):
    model_optim = optim.SGD(model.parameters(), lr=1e-4)
    
    for epoch in range(args.epochs):
        total_loss = torch.zeros(1)
        print("EPOCH", epoch)
        rand_idxs = np.random.permutation(len(train_data[0]))
        for i in tqdm(range(len(train_data[0])//args.batch_size)):
            # Predict on batch
            imgs = train_data[0][rand_idxs[i*args.batch_size:(i+1)*args.batch_size]]
            targets = train_data[1][rand_idxs[i*args.batch_size:(i+1)*args.batch_size]]
            preds = model(torch.tensor(imgs).cuda())
            
            # Update model
            loss = bce_loss_fn(preds, torch.tensor(targets, dtype=torch.float32).cuda())            
            loss.backward()
            model_optim.step()
            model_optim.zero_grad()
            total_loss += loss.cpu().detach().numpy()

        print("Total loss", total_loss)

    
    return model

def test_model(args, test_data, model):
    print("len(test_data[0])", len(test_data[0]))
    model = model.eval()
    y_preds = np.zeros((len(test_data[0]), 1))
    for i in tqdm(range(len(test_data[0])//args.batch_size)):
        imgs = test_data[0][i*args.batch_size:(i+1)*args.batch_size]
        targets = test_data[1][i*args.batch_size:(i+1)*args.batch_size]
        preds = model(torch.tensor(imgs).cuda())
        y_preds[i*args.batch_size:(i+1)*args.batch_size] = preds.cpu() >= args.decision_threshold
    
    return y_preds
    # test_f1_score = f1_score(y_trues, test_data[1])
    # print("F1 score of testing dataset", test_f1_score)
    # test_acc_score = np.sum(y_trues == test_data[1]) / len(y_trues)
    # print("Accuracy of testing dataset", test_acc_score)

    # return test_f1_score, test_acc_score