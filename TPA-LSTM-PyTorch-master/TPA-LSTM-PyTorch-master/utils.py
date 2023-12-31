"""
   Contains all the utility functions that would be needed
   1. _normalized
   2. _split
   3._batchify
   4. get_batches
   """


import torch
import numpy as np;
from torch.autograd import Variable


def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.) / (len(x)))


class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, cuda, horizon, window, normalize=2):
        self.cuda = cuda;
        self.window_length = window;
        self.horizon = horizon
        fin = open(file_name);
        column_indices = np.arange(start=1, stop=24)  # Replace 'num_columns' with actual number of columns
        self.original_data = np.loadtxt(fin, delimiter=",", skiprows=1, usecols=column_indices)
        self.normalized_data = np.zeros(self.original_data.shape);
        self.original_rows, self.original_columns = self.normalized_data.shape;
        self.normalize = 2
        self.scale = np.ones(self.original_columns);
        self._normalized(normalize);

        #after this step train, valid and test have the respective data, split from original_data according to the ratios
        self._split(int(train * self.original_rows), int((train + valid) * self.original_rows), self.original_rows);

        self.scale = torch.from_numpy(self.scale).float();
        list_of_tensors = [self.test[1][:, i, :] for i in range(self.test[1].size(1))]
        list_of_tmp = [list_of_tensors[i] * self.scale.expand(list_of_tensors[i].size(0), self.original_columns) for i in range(self.test[1].size(1))];
        if self.cuda:
            self.scale = self.scale.cuda();
        self.scale = Variable(self.scale);

        #rse and rae must be some sort of errors for now, will come back to them later
        self.list_of_rse = [normal_std(list_of_tmp[i]) for i in range(self.test[1].size(1))];
        # 计算list_of_rse的均值并添加到列表末尾
        mean_rse = sum(self.list_of_rse) / len(self.list_of_rse)
        self.list_of_rse.append(mean_rse)
        self.list_of_rae = [torch.mean(torch.abs(list_of_tmp[i] - torch.mean(list_of_tmp[i]))) for i in range(self.test[1].size(1))];
        # 计算list_of_rae的均值并添加到列表末尾
        mean_rae = sum(self.list_of_rae) / len(self.list_of_rae)
        self.list_of_rae.append(mean_rae)

    def _normalized(self, normalize):
        # normalized by the maximum value of entire matrix.

        if (normalize == 0):
            self.normalized_data = self.original_data

        if (normalize == 1):
            self.normalized_data = self.original_data / np.max(self.original_data);

        # normalized by the maximum value of each row(sensor).
        if (normalize == 2):
            for i in range(self.original_columns):
                self.scale[i] = np.max(np.abs(self.original_data[:, i]));
                self.normalized_data[:, i] = self.original_data[:, i] / np.max(np.abs(self.original_data[:, i]));

    def _split(self, train, valid, test):

        train_set = range(self.window_length + self.horizon - 1, train);
        valid_set = range(train, valid);
        test_set = range(valid, self.original_rows);
        self.train = self._batchify(train_set, self.horizon);
        self.valid = self._batchify(valid_set, self.horizon);
        self.test = self._batchify(test_set, self.horizon);

    def _batchify(self, idx_set, horizon):

        n = len(idx_set);
        X = torch.zeros((n, self.window_length, self.original_columns));
        Y = torch.zeros((n,horizon, self.original_columns));

        for i in range(n):
            end = idx_set[i] - horizon + 1;
            start = end - self.window_length;
            X[i, :, :] = torch.from_numpy(self.normalized_data[start:end, :]);
            Y[i,:, :] = torch.from_numpy(self.normalized_data[end:end+horizon, :]);

        """
            Here matrix X is 3d matrix where each of it's 2d matrix is the separate window which has to be sent in for training.
            Y is validation.           
        """
        return [X, Y];

    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt];
            Y = targets[excerpt];
            if (self.cuda):
                X = X.cuda();
                Y = Y.cuda();
            yield Variable(X), Variable(Y);
            start_idx += batch_size