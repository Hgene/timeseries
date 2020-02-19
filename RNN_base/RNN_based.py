import torch 
import torch.nn as nn

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
    
    

