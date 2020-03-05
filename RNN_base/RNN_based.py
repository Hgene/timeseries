import torch 
import torch.nn as nn


class MLSTM(nn.Module):
    '''LSTM for univaritate time series data, many-to-one version
    input_dim : dimension of input data
    hidden_layer : dimension of hidden layer
    output_dim : dimension of output/prediction values
    '''

    def __init__(self, input_dim=100, hidden_dim=3, sequence_length=7, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = sequence_length
        self.lstm = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)  # .to(device) when you use gpu
        self.fc = nn.Linear(hidden_dim, output_dim)  # .to(device) when you use gpu
        # (num_layers * num_directions, batch, hidden_size) whether batch_first=True or False
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_dim),  # .to(device),when you use gpu
                            torch.zeros(1, 1, self.hidden_dim))  # .to(device)) when you use gpu


    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)
        lstm_out = lstm_out.detach()
        predictions = self.fc(lstm_out[:, -1, :].view(1, -1))

        return predictions




class LSTM(nn.Module):
    '''LSTM for univaritate time series data, many-to-one version
    input_dim : dimension of input data
    hidden_layer : dimension of hidden layer
    output_dim : dimension of output/prediction values
    '''
    def __init__(self, input_dim=1, hidden_layer=10, output_dim=1):
        super().__init__()
        self.hidden_layer = hidden_layer
        self.lstm = nn.LSTM(input_dim, hidden_layer)
        self.linear = nn.Linear(hidden_layer, output_dim)

        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer),
                            torch.zeros(1, 1, self.hidden_layer))


    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


