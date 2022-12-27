import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

def create_sequences(logits, add_one=True):
    seq = []
    seq_lens = []
    tgts = []

    # Iterate over all subsequences in batch
    for j in reversed(range(logits.shape[1] - 1)):
        for i in range(logits.shape[0]):
            # Add input sequence
            seq.append(logits[i, 0:j+1])

            # Add the sequence length
            if add_one:
                seq_lens.append(j + 2)
            else:
                seq_lens.append(j + 1)
            
            
            # Create the targets for the discriminator
            tgts.append(logits[i, j+1])

    # Create the last for only state given as input
    for i in range(logits.shape[0]):
        seq.append(torch.zeros_like(logits[i, 0:1]))
        seq_lens.append(1)
        tgts.append(logits[i, 0])

    # Pad the sequences to the same length
    seq = pad_sequence(seq, batch_first=True)

    return seq, seq_lens, tgts

def create_disc_train_data(task_ids, logits, config):
    # Create the dataset
    seq, seq_len, tgts = create_sequences(logits)
    seq = seq.to(config.device)

    # Create the targets
    tgts = torch.stack(tgts).to(config.device)
    # tgts = tgts.softmax(dim=1)
    # Need to create the corrrect indices so only 
    data_idxs = torch.tensor(seq_len).to(config.device) - 1

    # Repeat the images and make them in the correct order
    task_ids = task_ids.repeat((logits.shape[1], 1))
    
    return seq, seq_len, tgts, data_idxs, task_ids

def extract_lstm_out(x, data_idxs):
    return x[torch.arange(x.shape[0]), data_idxs].squeeze()

def compute_entropy_loss(config, logits):
    tgt = torch.zeros(config.routing_depth, config.num_modules).to(config.device)
    for idx, valid_idxs in config.valid_mbs.items():
        tgt[idx, torch.tensor(valid_idxs)] = 1/config.unique_hidden_modules

    avg_logits = logits.mean(dim=0)
    
    return F.cross_entropy(avg_logits, tgt)
    




# print_iter = 0
# def balance_loss(logits, config, max_prob=0.35, min_prob=0.001, min_temp=0.5):
#     global print_iter
#     distr = F.softmax(logits/min_temp, dim=-1).sum(axis=[0,1])/logits.shape[0]
#     print_iter = print_iter + 1
#     # print("distr", distr)
#     # print("distr_clipping", distr_clipping, "\n")


#     # if distr_clipping.shape[0] > 0:
#     tgt = torch.zeros_like(distr)
#     tgt[distr > max_prob] = max_prob
#     tgt[distr <= max_prob] = torch.clip(distr[distr <= max_prob], min=min_prob)
#     # print("distr", distr)
#     # print("tgt", tgt)
#     total_entropy_loss = F.binary_cross_entropy(distr, tgt.detach())
#     if print_iter % 90 == 0:
#         print(F.softmax(logits/min_temp, dim=-1)[:, -1, -config.num_modules:])
#         print("distr", distr)
#         print("tgt", tgt)
#     # else:
#     #     total_entropy_loss = 0

#     # distr_clipping_min = distr[distr < min_prob]
#     # if distr_clipping_min.shape[0] > 0:
#     #     total_entropy_loss = total_entropy_loss + F.binary_cross_entropy(
#     #         distr_clipping_min,
#     #         torch.tensor([min_prob]).repeat(distr_clipping_min.shape[0]).to(config.device)) * 0.1

    

#     #total_entropy_loss = torch.square(torch.clip(distr - 0.5, min=0)).mean()
#     #total_entropy_loss += torch.square(torch.clip(0.01 - distr , min=0)).mean()
#     return total_entropy_loss

    






    