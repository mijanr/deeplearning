import torch    
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_ch:int=1, num_classes:int=10):
        """
        Convolutional Neural Network

        Parameters
        ----------
        in_ch : int
            Number of input channels
        num_classes : int
            Number of classes
        """
        super(CNN, self).__init__()
        self.seq = nn.Sequential(

            # conv block
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.25),

            # fc block
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )
        
        
    def forward(self, x):
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        torch.Tensor
            Output tensor
        """
        return self.seq(x)
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model parameters
    in_ch = 1
    num_classes = 10

    # create model
    model = CNN(in_ch, num_classes).to(device)

    # test the model
    x = torch.randn(64, in_ch, 28, 28).to(device)
    y = model(x)
    print(y.shape)