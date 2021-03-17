import os
import numpy as np

class DataLoader:
    def __init__(self, args):
        self.args = args

    def load_mnist(self):
        train_data = self.load_npy(self.args.train_data)
        test_data =  self.load_npy(self.args.test_data)
        return train_data, test_data

    def load_npy(self, filename):
        # Load testing data
        with open(filename, 'rb') as f:
            fsz = os.fstat(f.fileno()).st_size
            print(fsz)
            data = np.load(f, allow_pickle=True)
            while f.tell() < fsz:
                data = np.vstack((data, np.load(f, allow_pickle=True)))
        return data