流程工业控制系统时序数据预测任务
赵思洋,朱佳怡,李天蔚
2023 年 12 月 17 日


1	分工
赵思洋负责模型的学习和修改,并负责论文和PPT的代码部分.朱佳怡负责论文的编写. 李天蔚负责PPT.

2	引言
PV_{i}和 OP_{i}列表示第 i 个回路中对应的过程变量和控制变量。DV_{1- 5}列是工段的干扰变量，23个变量预测9个过程变量。使用过去 30 个时间点预测未来 5 个时间点，采用一个模型迭代预测 5步，并将得到的结果与五个模型分别预测进行比较。
提取每次预测的Y值9个过程变量与真实值的14个其他变量进行拼接,在31-35时间段对学习数据X未知的情况下将X进行似然假设处理,即预测第35次时,X的6-30是真实值,31-34来自前4次的预测值.该方法的后4个准确度依赖第1次的准确度.即corr^5.
本报告通过阅读学习参考文献 [3]，修改开源代码，使其能够处理本任务所提供的数据集。本文将在第 3 节详细描述算法模型设计,第 4节为结果与讨论。
 


3	相关工作
该研究（石舜尧等人）详细介绍了一种创新的时间模式注意力机制，用于多变量时间序列预测。该模型通过提取时间不变的时间模式，并利用频域信息进行多变量预测。实验证明，该模型在多个真实世界任务中表现出色。传统的注意力机制通常在每个时间步骤上检查信息并选择相关信息来辅助输出生成，但无法捕捉跨多个时间步骤的时间模式。为了解决这个问题，研究提出使用一组滤波器来提取时间不变的时间模式，类似于将时间序列数据转换为其“频域”。具体而言，引入了卷积神经网络（CNN）来从每个单  独的变量中提取时间模式信息。这些滤波器能够捕捉到时间序列数据中的时间模式，并将其转化为“频域”信息。通过这种方式，模型能够更好地捕 捉到时间序列数据中的长期依赖关系，并提高预测的准确性。然后，研究提出了一种新颖的注意力机制，用于选择相关的时间序列，并利用其频域信息进行多变量预测。根据实验结果，该模型在多个实验中表现出色。在典型的多变量时间序列数据集上，该模型在预测准确性方面优于其他方法，包括LSTNet-Skip 和 LSTNet-Attn 等先进方法。无论是周期性还是非周期性的数据集，该模型都能取得最佳的性能。此外，该模型还能处理各种规模的数据集，从最小的 534KB 的汇率数据集到最大的 172MB 的太阳能数据集。
对于所需处理的工业数据集，由于其中包含大量的传感器读数，包括过
程变量、控制变量和干扰变量，使用论文中所介绍的方法进行处理具有以下优势：1. 多元预测：该方法适用于处理多元时间序列数据，能够同时考虑多个传感器的读数，从而更准确地进行预测。2. 长期依赖建模：该方法采用了循环神经网络（RNN）和注意力机制，能够捕捉到时间序列数据中的长期依赖关系，从而更好地构建模型和进行数据预测。3. 时间模式提取：该方法通过使用一组滤波器来提取时间不变的时间模式，类似于将时间序列数据转换为频域信息。这样可以更好地理解和表达数据中的重要特征。4. 高性能： 根据文中的实验结果显示，该方法在多个真实世界任务中表现出色，达到了最先进的性能水平。因此，使用该方法处理这个工业数据集可以获得准确可靠的预测结果。
 


4	模型修改
4.1	数据处理
Data_utility 函数处理 jl_data_train.csv
self.original_data = np.loadtxt(fin, delimiter=",", skiprows=1, usecols=column_indices)
预测未来五步，获取Y为list
Y = torch.zeros((n,horizon, self.original_columns));
Y[i,:, :] = torch.from_numpy(self.normalized_data[end:end+horizon, :]);

4.2	神经网络
激活函数
原定使用sigmod作为输出激活函数无法输出负数，改为使用tanh
parser.add_argument('--output_fun', type=str, default='tanh')
注意力加权激活函数sigmo改为softmax
alpha = torch.nn.functional.softmax(scoring_function, dim=1)  # 使用softmax替代sigmoid


修改网络的输出特征数为9:
Final_matrix:
self.final_matrix = nn.Parameter(
    torch.ones(args.batch_size, 9, self.hidden_state_features, requires_grad=True)) #, device='cuda'
result = torch.bmm(self.final_matrix, h_intermediate)
hw网络:
if (self.hw > 0):
    z = x[:, -self.hw:, :9];
    z = z.permute(0, 2, 1).contiguous().view(-1, self.hw);
    z = self.highway(z);
    z = z.view(-1, 9);
    res = final_result + z;

尝试使用过全连接层最后处理,效果差
# 将res输入到全连接层
#res = self.fc(res)
# 应用激活函数
#res = F.tanh(res)
4.3	主函数
修改train和evaluate,使其能够30出5:
使用一个模型在1-5次中,从第2次预测起,使用前一次的output的前九个特征(过程变量)与Y的其他特征拼接,组成似然X预测下一个horizon.

参数:
parser.add_argument('--window', type=int, default=30,
                    help='window size')
parser.add_argument('--horizon', type=int, default=5)
同时修改hw网络回溯窗口,使得-5次似然值得到强化处理
parser.add_argument('--highway_window', type=int, default=5,
                    help='The window size of the highway component')

运行逻辑:
def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size):
    model.eval();
    total_loss = [0] * 6
    total_loss_l1 = [0] * 6
    n_samples = 0;
    # 初始化predict和test为长度为5的列表，每个元素都为None
    predict = [None] * 5
    Ytest = [None] * 5

    for X, Y in data.get_batches(X, Y, batch_size, False):
        with torch.no_grad():  # 确保在评估时不会计算梯度
            output = model(X)[:, :9]  # 获取模型前九个维度的输出
            if predict[0] is None:
                predict[0] = output;
                Ytest[0] = Y[:, 0, :9];
            else:
                predict[0] = torch.cat((predict[0], output));
                Ytest[0] = torch.cat((Ytest[0], Y[:, 0, :9]));
        scale = data.scale.expand(output.size(0), -1)[:, :9]  # 确保 scale 张量与 output 的批次大小匹配
        loss = evaluateL2(output * scale, Y[:, 0, :9] * scale)
        total_loss[0] += loss.item()  # 累加每一步的损失

        loss = evaluateL1(output * scale, Y[:, 0, :9] * scale)
        total_loss_l1[0] += loss.item()  # 累加每一步的损失
        for step in range(5 - 1):  # 已经有了1次初始预测，所以再做4次
            # 准备下一步的输入
            X_part = X[:, 1:, :]
            Y_part = Y[:, step, 9:]

            # 按最后一维拼接这两部分
            trick = torch.cat((output, Y_part), dim=1)
            trick = trick.unsqueeze(1)
            X = torch.cat((X_part, trick), dim=1)
            # 做出新的预测
            output = model(X)[:, :9]
            # 检查predict和test的当前元素是否为None，并进行赋值或拼接
            if predict[step+1] is None:
                predict[step+1] = output
                Ytest[step+1] = Y[:, step+1, :9]
            else:
                predict[step+1] = torch.cat((predict[step+1], output), dim=0)
                Ytest[step+1] = torch.cat((Ytest[step+1], Y[:, step+1, :9]), dim=0)

            # 累积损失
            loss = evaluateL2(output * scale, Y[:, step + 1, :9] * scale)
            total_loss[step+1] += loss.item()  # 累加每一步的损失
            loss = evaluateL1(output * scale, Y[:, step + 1, :9] * scale)
            total_loss_l1[step+1] += loss.item()  # 累加每一步的损失


        n_samples += (output.size(0) * output.size(1));

    avg_loss = sum(total_loss) / n_samples
    avg_loss_l1 = sum(total_loss_l1) / n_samples

    total_loss[-1] += avg_loss  # 累加平均损失
    total_loss_l1[-1] += avg_loss_l1  # 累加平均损失

    # 计算相对误差
    list_of_rse = [math.sqrt(loss_L2 / n_samples) / rse for loss_L2, rse in zip(total_loss, data.list_of_rse)]
    list_of_rae = [loss_L1 / n_samples / rae for loss_L1, rae in zip(total_loss_l1, data.list_of_rae)]

    # 假设 predict 和 Ytest 是长度为5的列表，每个元素都是一个Tensor
    predict = [p.detach().cpu().numpy() for p in predict if p is not None]
    Ytest = [y.detach().cpu().numpy() for y in Ytest if y is not None]

    # 现在 predict_tensors 和 Ytest_tensors 是包含numpy数组的列表

    #print(predict.shape, Ytest.shape)

    # 计算每个步骤的相关性
    correlations = []
    for step in range(5):
        if predict[step] is not None and Ytest[step] is not None:
            sigma_p = predict[step].std(axis=0)
            sigma_g = Ytest[step].std(axis=0)
            mean_p = predict[step].mean(axis=0)
            mean_g = Ytest[step].mean(axis=0)
            index = (sigma_g != 0)
            correlation = ((predict[step] - mean_p) * (Ytest[step] - mean_g)).mean(axis=0) / (sigma_p * sigma_g)
            correlation = (correlation[index]).mean() if np.any(index) else 0
            correlations.append(correlation)
        else:
            correlations.append(None)
    mean_corr = sum(correlations) / len(correlations)
    correlations.append(mean_corr)
    return list_of_rse, list_of_rae, correlations,predict,Ytest;
:以列表形式进行评估和训练:
def train(data, X, Y, model, criterion, optim, batch_size):  # X is train set, Y is validation set, data is the whole data
    model.train();
    total_losses = [0] * 6  # 初始化一个长度为6的列表，用于存储每步的损失和最后的平均损失
    n_samples = 0;
    for X, Y in data.get_batches(X, Y, batch_size, True):
        #print(Y)
        model.zero_grad();
        output = model(X)[:, :9]  # 获取模型前九个维度的输出
        # 切片 data.scale 以匹配前9个维度的输出
        scale = data.scale.expand(output.size(0), -1)[:, :9]  # 确保 scale 张量与 output 的批次大小匹配
        list_of_loss = []
        loss = criterion(output * scale, Y[:, 0, :9] * scale)
        list_of_loss.append(loss)
        total_losses[0] += loss.detach().item()  # 累加每一步的损失
        for step in range(5 - 1):  # 已经有了1次初始预测，所以再做4次
            # 准备下一步的输入
            X_part=X[:,1:,:]
            Y_part = Y[:, step, 9:]

            # 按最后一维拼接这两部分
            trick = torch.cat((output, Y_part), dim=1)
            trick=trick.unsqueeze(1)
            X=torch.cat((X_part, trick), dim=1)
            # 做出新的预测
            output = model(X)[:, :9]
            # 累积损失
            loss = criterion(output * scale, Y[:, step + 1, :9] * scale)
            list_of_loss.append(loss)
            total_losses[step+1] += loss.detach().item()  # 累加每一步的损失


        # 取5步预测的平均损失
        avg_loss = torch.stack(list_of_loss).mean()
        avg_loss.backward();
        #list_of_loss[].backward();
        optim.step()
        total_losses[-1] += avg_loss.item()  # 累加平均损失
        n_samples += (output.size(0) * output.size(1))
        average_losses = [total_loss / n_samples for total_loss in total_losses]

    return average_losses
    return 1




评估指标使用列表记录.在1-5次中,从第2次预测起,使用前一次的output的前九个特征(过程变量)与Y的其他特征拼接,组成似然X:
# 按最后一维拼接这两部分
trick = torch.cat((output, Y_part), dim=1)
trick=trick.unsqueeze(1)
X=torch.cat((X_part, trick), dim=1)
# 做出新的预测
output = model(X)[:, :9]
train使用avg_loss张量进行反向传播:
# 取5步预测的平均损失
avg_loss = torch.stack(list_of_loss).mean()
avg_loss.backward();
#list_of_loss[].backward();
输出predicate和Ytest展示训练效果:
plt.figure(figsize=(15, 10))  # 设置总图表的大小
# 现在 test_predict 和 test_Ytest 是包含所有时间步长预测和真实值的列表
for i in range(len(test_predict)):
    plt.subplot(3, 2, i + 1)  # 创建子图
    plt.plot(test_predict[i][:, 1], 'r-', label='Predictions')  # 绘制预测值
    plt.plot(test_Ytest[i][:, 1], 'b-', label='Actual')  # 绘制实际值
    plt.title(f'Time step {i + 1} Predictions vs Actual')
    plt.xlabel('Sample index')
    plt.ylabel('Value')
    if i == 0:  # 只在第一个子图中显示图例
        plt.legend()

plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
plt.show()
5个模型分别预测:以列表形式:
# 首先创建Data_utility对象列表
data_utilities = [
    Data_utility(args.data, 0.6, 0.2, args.cuda, i, args.window, args.normalize) for i in range(1, 6)
]

# 使用列表推导来创建模型列表
models = [eval(args.model).Model(args, data) for data in data_utilities]
# 计算每个模型的参数数量并打印
for i, model in enumerate(models):
    nParams = sum([p.nelement() for p in model.parameters()])
    print(f'* number of parameters for model {i+1}: {nParams}')

# 创建每个模型的优化器
optims = [
    torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    for model in models
]
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
5	结果与讨论
一个模型渐进式迭代:第六行数据为平均值 
第五次的结果约等于第一次的五次方,基本符合预期.![屏幕截图 2023-12-17 171735](https://github.com/orz-c3/python-/assets/128114230/d570e679-9301-4494-9f16-d0402e4a577c)

五个模型分别预测:  ![屏幕截图 2023-12-17 174139](https://github.com/orz-c3/python-/assets/128114230/4e6d7823-2f21-4633-98fe-b1a4d6d71fdc)

由此可见,一个模型渐进迭代学习速度上远远快于五个模型,同时随着horizon增加,退化率处在可接受范围内.
下面是渐进式学习的效果展示: 可见随着预测时间的后延,效果逐渐退化 ![屏幕截图 2023-12-17 171624](https://github.com/orz-c3/python-/assets/128114230/79ea7008-9325-45ec-b2f4-89b2f2e75042)

