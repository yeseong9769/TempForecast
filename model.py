import torch.nn as nn

class WeatherPredicter(nn.Module):
    """
    Args:
        input_shape (int): 입력 데이터의 feature 개수.
        output_shape (int): 예측 대상의 개수 (forecast horizon).
        hidden_size (int): LSTM 계층의 은닉층 크기. 기본값은 100.
        n_layers (int): LSTM 계층의 개수. 기본값은 2.
        dropout_p (float): 드롭아웃 확률. 기본값은 0.2.
    """
    def __init__(self, input_shape, output_shape, hidden_size=100,  n_layers=2, dropout_p=0.2):
        self.input_shape = input_shape
        self.hidden_size = hidden_size
        self.output_shape = output_shape
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        super().__init__()

        self.lstm = nn.LSTM(input_shape, 
                            hidden_size, 
                            n_layers, 
                            dropout=dropout_p, 
                            batch_first=True)
        
        self.seq = nn.Sequential(
            nn.Linear(hidden_size, 30),
            nn.Linear(30, output_shape)
        )

    def forward(self, x):
        # x: (batch_size, seq_length, input_shape)
        z, _ = self.lstm(x)
        # z: (batch_size, seq_length, hidden_size)
        z = z[:, -1, :]
        # z: (batch_size, hidden_size)
        y = self.seq(z)

        return y