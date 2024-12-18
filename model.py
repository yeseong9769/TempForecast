import torch.nn as nn

class WeatherPredicter(nn.Module):
    def __init__(self, input_shape, output_shape, hidden_size, n_layers=2, dropout_p=0.2):
        super().__init__()
        
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.output_shape = output_shape
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        # LSTM Layer
        self.lstm = nn.LSTM(input_shape, 
                            hidden_size, 
                            n_layers, 
                            dropout=dropout_p, 
                            batch_first=True)
        
        # Fully Connected Layer
        self.seq = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Linear(hidden_size // 2, output_shape)
        )

    def forward(self, x):
        # x: (batch_size, seq_length, input_shape)
        z, _ = self.lstm(x)
        # z: (batch_size, seq_length, hidden_size)
        z = z[:, -1, :]
        # z: (batch_size, hidden_size)
        y = self.seq(z)

        return y