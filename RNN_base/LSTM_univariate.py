import torch
import torch.nn as nn

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from RNN_based import LSTM


def create_inout_sequences(input_data ,tw):
    '''data preprocessing, create (input_train, output_train) sequences
        input_data : data set you want to train,
        tw : time window size
        output : (input_train, output_train) by time window size(tw)
        '''
    inout_seq =[]
    L = len(input_data)
    for i in range( L -tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw: i+tw+1]  # to make it list shape
        inout_seq.append((train_seq ,train_label))

    return inout_seq


def preprocessing(train_data):
    '''data preprocessing, normalize dataset
    train_data : data set you want to normalized
    output : scaler - information of normalize
             normalized_torch - normalized data
    '''
    scaler = MinMaxScaler(feature_range=(-1, 1))
    normalized = scaler.fit_transform(train_data.reshape(-1, 1))
    normalized_torch = torch.FloatTensor(normalized).view(-1)

    return scaler, normalized_torch


def mse(true, pred):
    '''caculate mse loss '''
    return np.sqrt(np.sum((true-pred)**2)/len(true))


def train(train_inout, epochs = 150, lr = 0.0001):
    '''model generation and train using (input_train,output_train) data set
    train_inout : (input_train,output_train) dataset
    epochs : number of epochs
    lr : learning rate
    '''
    model = LSTM()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    print(model)

    #Train model

    for i in range(epochs):
        for seq, labels in train_inout:
            optimizer.zero_grad()
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer),
                                 torch.zeros(1, 1, model.hidden_layer))

            y_pred = model(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()
            optimizer.step()

        if i % 25 == 1:
            print('epoch: {}, loss: {}'.format(i, single_loss.item()))

    print('epoch: {}, loss: {}'.format(i, single_loss.item()))

    return model


def test(model, scaler, test_inputs, test_len):
    '''trained model test, predict future value using input_train[-train_window:] data set
    model : trained model
    scaler : model trained by normalized data, so it need to inverse transformation
    test_inputs : input_train[-train_window:] data set
    test_len : length of prediction you want
    '''
    model.eval()
    train_window = len(test_inputs)

    for i in range(test_len):
        seq = torch.FloatTensor(test_inputs[-train_window:])
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer),
                           torch.zeros(1, 1, model.hidden_layer))
            test_inputs.append(model(seq).item())

    #predicted value : test_inputs[fut_pred:] we have to inverse transformation because of normalization
    preds = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1,1) )
    print(preds)

    return preds


def main():

    # load data form seaborn package
    flight_data = sns.load_dataset("flights")
    all_data = flight_data['passengers'].values.astype(float)
    print(all_data)

    # choose test data size and data preprocessing
    test_len = 15
    train_data = all_data[:-test_len]
    test_data = all_data[-test_len:]

    train_window = 12
    scaler, normalized_torch = preprocessing(train_data)
    train_inout = create_inout_sequences(normalized_torch, train_window)

    #model train
    epochs = 170
    lr = 0.001
    model = train(train_inout, epochs , lr)

    #test data preprocessing
    inputs_for_pred = normalized_torch[-train_window:].tolist()

    #test your data
    preds = test(model, scaler, inputs_for_pred, test_len)
    print('Prediction Error is : {}'.format(mse(test_data, preds)))

    #plot prediction values and actual value
    x = np.arange(len(train_data), len(train_data)+test_len , 1)
    plt.title('Month vs Passenger')
    plt.ylabel('Total Passengers')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(all_data)
    plt.plot(x,preds)
    plt.show()

if __name__ == '__main__':
    main()