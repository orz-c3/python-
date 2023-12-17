"""THE DRIVER CLASS TO RUN THIS CODE"""

"""FUTURE SCOPE, ADD ARGUMENTS AS NEEDED"""

import matplotlib.pyplot as plt
import argparse
import math
import time

import torch
import torch.nn as nn

from models import TPA_LSTM_Modified
from utils import *
import numpy as np;
import importlib
parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default="data/jl_data_train.csv",
                    help='location of the data file')
#, required=True
parser.add_argument('--model', type=str, default='TPA_LSTM_Modified',
                    help='')
parser.add_argument('--hidden_state_features', type=int, default=12,
                    help='number of features in LSTMs hidden states')
parser.add_argument('--num_layers_lstm', type=int, default=1,
                    help='num of lstm layers')
parser.add_argument('--hidden_state_features_uni_lstm', type=int, default=1,
                    help='number of features in LSTMs hidden states for univariate time series')
parser.add_argument('--num_layers_uni_lstm', type=int, default=1,
                    help='num of lstm layers for univariate time series')
parser.add_argument('--attention_size_uni_lstm', type=int, default=10,
                    help='attention size for univariate lstm')
parser.add_argument('--hidCNN', type=int, default=10,
                    help='number of CNN hidden units')
parser.add_argument('--hidRNN', type=int, default=100,
                    help='number of RNN hidden units')
parser.add_argument('--window', type=int, default=30,
                    help='window size')
parser.add_argument('--CNN_kernel', type=int, default=1,
                    help='the kernel size of the CNN layers')
parser.add_argument('--highway_window', type=int, default=24,
                    help='The window size of the highway component')
parser.add_argument('--clip', type=float, default=10.,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=30,
                    help='upper epoch limit') #30
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=54321,
                    help='random seed')
parser.add_argument('--gpu', type=int, default=None)
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model/model.pt',
                    help='path to save the final model')
parser.add_argument('--cuda', type=str, default=False)
parser.add_argument('--optim', type=str, default='sgd')
parser.add_argument('--lr', type=float, default=1e-05)
parser.add_argument('--momentum', type=float, default=0.9)#改为0.9
parser.add_argument('--horizon', type=int, default=1)
parser.add_argument('--skip', type=float, default=24)
parser.add_argument('--hidSkip', type=int, default=5)
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--output_fun', type=str, default='tanh')#改成tanh
args = parser.parse_args()

def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval();
    total_loss = 0;
    total_loss_l1 = 0;
    n_samples = 0;
    predict = None;
    test = None;
    Y=Y[:,:9];

    for X, Y in data.get_batches(X, Y, batch_size, False):
        output = model(X)[:,:9];
        if predict is None:
            predict = output;
            test = Y;
        else:
            predict = torch.cat((predict, output));
            test = torch.cat((test, Y));
        scale = data.scale.expand(output.size(0), -1)[:, :9]  # 确保 scale 张量与 output 的批次大小匹配
        total_loss += evaluateL2(output * scale, Y * scale).data


        total_loss_l1 += evaluateL1(output * scale, Y * scale).data
        n_samples += (output.size(0) * output.size(1));
    rse = math.sqrt(total_loss / n_samples) / data.rse
    rae = (total_loss_l1 / n_samples) / data.rae

    predict = predict.data.cpu().numpy();
    Ytest = test.data.cpu().numpy();
    sigma_p = (predict).std(axis=0);
    sigma_g = (Ytest).std(axis=0);
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0);
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / (sigma_p * sigma_g);
    correlation = (correlation[index]).mean();
    return rse, rae, correlation,predict,Ytest;


def train(data, X, Y, model, criterion, optim, batch_size):  # X is train set, Y is validation set, data is the whole data
    model.train();
    total_loss = 0;
    n_samples = 0;
    for X, Y in data.get_batches(X, Y, batch_size, True):
        #print(Y)
        model.zero_grad();
        output = model(X)[:,:9];
        scale = data.scale.expand(output.size(0), -1)[:, :9]  # 确保 scale 张量与 output 的批次大小匹配
        loss = criterion(output * scale, Y[:,:9] * scale);
        loss.backward();
        grad_norm = optim.step();
        total_loss += loss.data;
        n_samples += (output.size(0) * output.size(1));
    return total_loss / n_samples
    return 1




# 首先创建Data_utility对象列表
data_utilities = [
    Data_utility(args.data, 0.6, 0.2, args.cuda, i, args.window, args.normalize) for i in range(1, 6)
]

# 使用列表推导来创建模型列表
models = [eval(args.model).Model(args, data) for data in data_utilities]

# 如果args.cuda为True，将每个模型转移到CUDA
if args.cuda:
    for model in models:
        model.cuda()


#print(dict(model.named_parameters()))
if args.L1Loss:
    criterion = nn.L1Loss(size_average=False);
else:
    criterion = nn.MSELoss(size_average=False);
evaluateL2 = nn.MSELoss(size_average=False);
evaluateL1 = nn.L1Loss(size_average=False)
if args.cuda:
    criterion = criterion.cuda()
    evaluateL1 = evaluateL1.cuda();
    evaluateL2 = evaluateL2.cuda();
# 计算每个模型的参数数量并打印
for i, model in enumerate(models):
    nParams = sum([p.nelement() for p in model.parameters()])
    print(f'* number of parameters for model {i+1}: {nParams}')

# 创建每个模型的优化器
optims = [
    torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for model in models
]

# 初始化最佳验证分数列表，设置一个很高的初始值
best_vals = [10000000] * len(models)
best_val_avg = 10000000;

try:
    print('begin training');
    # 现在我们可以开始训练和验证循环
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()
        train_losses = []
        val_losses = []
        val_raes = []
        val_corrs = []

        # 训练并评估每个模型
        for data_utility, model, optim in zip(data_utilities, models, optims):
            train_loss = train(data_utility, data_utility.train[0], data_utility.train[1], model, criterion, optim,
                               args.batch_size)
            val_loss, val_rae, val_corr, val_predict, val_Ytest = evaluate(data_utility, data_utility.valid[0],
                                                                           data_utility.valid[1], model, evaluateL2,
                                                                           evaluateL1, args.batch_size)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_raes.append(val_rae)
            val_corrs.append(val_corr)

            print(
                '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f}'.format(
                    epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr))

        # 打印平均损失和指标
        print(
            '| end of epoch {:3d} | time: {:5.2f}s | avg_train_loss {:5.4f} | avg_valid rse {:5.4f} | avg_valid rae {:5.4f} | avg_valid corr  {:5.4f}'.format(
                epoch, (time.time() - epoch_start_time),
                sum(train_losses) / len(train_losses),
                sum(val_losses) / len(val_losses),
                sum(val_raes) / len(val_raes),
                sum(val_corrs) / len(val_corrs)
            )
        )
        print("\n")
        # 更新最佳验证损失
        for i, (val_loss, model) in enumerate(zip(val_losses, models)):
            if val_loss < best_vals[i]:
                # with open(args.save, 'wb') as f:
                #     torch.save(model, f)
                best_vals[i] = val_loss

        # 每5个epoch进行一次评估
        if epoch % 5 == 0:
            for i, (data_utility, model) in enumerate(zip(data_utilities, models)):
                test_acc, test_rae, test_corr, test_predict, test_Ytest = evaluate(data_utility, data_utility.test[0],
                                                                                   data_utility.test[1], model,
                                                                                   evaluateL2, evaluateL1,
                                                                                   args.batch_size)
                print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}".format(test_acc, test_rae, test_corr))
            print("\n")
except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')