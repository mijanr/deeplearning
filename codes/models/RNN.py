import torch
import torch.nn as nn

class RecurrentNN(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, num_layers:int=2, dropout:float=0.25):
        """
        Recurrent Neural Network

        Parameters
        ----------
        input_dim : int
            Number of input dimensions
        hidden_dim : int
            Number of hidden dimensions
        output_dim : int
            Number of output dimensions
        num_layers : int, optional
            Number of layers, by default 2
        dropout : float, optional
            Dropout rate, by default 0.25        
        """
        super(RecurrentNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # initialize hidden state with zeros
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(device)

        # forward pass
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out       

    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model parameters
    input_dim = 28
    hidden_dim = 128
    output_dim = 10
    num_layers = 2
    dropout = 0.25
    
    # create model
    model = RecurrentNN(input_dim, hidden_dim, output_dim, num_layers, dropout).to(device)
    print(model)
    
    # input
    x = torch.randn(64, 28, 28).to(device)
    print(x.shape)
    
    # forward pass
    out = model(x)
    print(out.shape)