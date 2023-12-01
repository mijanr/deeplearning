import torch
import torch.nn as nn

class FeedForwardNN(nn.Module):

    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int, dropout:float=None):
        """
        Initializes the FeedForwardNN model.

        Parameters
        ----------
        input_dim : int
            The dimension of the input tensor.
        hidden_dim : int
            The dimension of the hidden layer.
        output_dim : int
            The dimension of the output tensor.
        dropout : float, optional
            The dropout rate, by default None  (no dropout).
        """
        super(FeedForwardNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout

        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        """
        Performs a forward pass on the FeedForwardNN model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor.

        Returns
        -------
        torch.Tensor
            The output tensor.
        """
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # model parameters
    input_dim = 10
    hidden_dim = 20
    output_dim = 1
    dropout = 0.2

    # create model
    model = FeedForwardNN(input_dim, hidden_dim, output_dim, dropout).to(device)

    # test the model
    x = torch.rand(32, input_dim).to(device)
    y = model(x)
    print(y.shape)