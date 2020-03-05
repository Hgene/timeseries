import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable

from RNN_based import MLSTM


def build_inout_seq(input_data, seq_length):
    '''data preprocessing, create (input_train, label_train) sequences
        input_data : data set you want to train,
        seq_length : time window size, same as length of sequences, same as number of layers
        return value(inout_seq) : (input_train, output_train) by window size same as seqeunce length(seq_length)
        '''
    inout_seq = []
    L = len(input_data)
    for i in range(L - seq_length):
        train_seq = input_data[i:i + seq_length]
        train_label = input_data[i + seq_length: i + seq_length + 1,
                      3]  # to make it list shape 4th column is value of CO2
        inout_seq.append((train_seq, train_label))

    return inout_seq


def main():
    torch.manual_seed(0)

    #data load
    trainset = pd.read_csv('./dataset/occupancy/train.txt').values[:, 1:6]
    testset = pd.read_csv('./dataset/occupancy/test1.txt').values[:, 1:6]

    #scaling data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(trainset)
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    norm_train = torch.FloatTensor(scaler.transform(trainset))  # .to(device)
    norm_test = torch.FloatTensor(scaler.transform(testset))  # .to(device)

    print('Train data shape : {}'.format(norm_train.shape))
    print('Test data shape : {}'.format(norm_test.shape))

    #build dataset using sequence length, we suppose last 7 days data affect to CO2 level of next day
    seq_length = 7
    train_inout = build_inout_seq(norm_train, seq_length)

    input_dim = 5
    hidden_dim = 3
    sequence_len = 7 # we suppose last 7 days data affect to CO2 level of next day
    output_dim = 1
    lr = 0.0005
    epochs = 15

    model = MLSTM(input_dim, hidden_dim, sequence_len, output_dim)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    print(model)

    train_preds = []
    train_acts = []
    # Train model using Multivariate LSTM, many to one version
    for i in range(epochs):
        for seq, label in train_inout:
            # Input: (batch, seq_len, input_size) when batch_first=True
            seq = Variable(seq).view(1, seq_length, -1)  # add `.to(device)` when you use gpu
            label = Variable(label).view(1, -1)  # add `.to(device)` when you use gpu

            output = model(seq)
            loss = loss_function(output, label)

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            train_preds.append(output.detach())
            train_acts.append(label.detach())

        print('epoch: {}, loss: {}'.format(i, loss.item()))

    plt.title("Train data prediction result, Normalize version, epoch {} ".format(1))
    plt.xlabel("x")
    plt.ylabel("CO2")
    plt.plot(range(len(train_acts[:8136])), train_acts[:8136], 'b-', range(len(train_preds[:8136])), train_preds[:8136], 'r-')
    plt.savefig('train_result_epoch{}.png'.format(1))
    plt.clf()

    plt.title("Train data prediction result, Normalize version, epoch {} ".format(epochs))
    plt.xlabel("x")
    plt.ylabel("CO2")
    plt.plot(range(len(train_acts[-8136:])), train_acts[-8136:], 'b-', range(len(train_preds[-8136:])), train_preds[-8136:],
             'r-')
    plt.savefig('train_result_epoch{}.png'.format(epochs))
    plt.clf()

    torch.save(model.state_dict(), 'lstm.ckpt')


    # Test data preprocessing
    test_inout = build_inout_seq(norm_test, seq_length)

    # Test the model
    preds = []
    actuals = []
    with torch.no_grad():
        total_err = 0

        for seq, label in test_inout:
            seq = Variable(seq).view(1, seq_length, -1)  # .to(device)
            label = Variable(label).view(1, -1)  # .to(device)
            actuals.append(label.detach())

            output = model(seq)  # .detach()
            preds.append(output.detach())
            error = loss_function(output, label)  # .view(1, 1)
            total_err += error

        print('Test Error of the model : {} '.format(total_err))

    plt.title("Test data prediction result, Normalize version")
    plt.xlabel("x")
    plt.ylabel("CO2")
    plt.plot(range(len(actuals)), actuals, 'b-', range(len(preds)), preds, 'r-')
    plt.savefig('test_result_norm.png')
    plt.clf()
    #plt.show()

    actuals = testset[seq_length:, 3]
    preds_pre = np.array(norm_test[seq_length:])
    preds_pre[:, 3] = preds
    preds = scaler.inverse_transform(preds_pre)[:, 3]

    plt.title("Test data prediction result, Actual Value")
    plt.xlabel("x")
    plt.ylabel("CO2")
    plt.plot(range(len(actuals)), actuals, 'b-', range(len(preds)), preds, 'r-')
    plt.savefig('test_result.png')
    plt.show()


if __name__ == '__main__':
    main()