# File: data/src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FloodCNNLSTM(nn.Module):
    def __init__(self, n_features=21, seq_length=24, hidden_size=128,
                 n_lstm_layers=2, dropout=0.3):
        super(FloodCNNLSTM, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout_cnn = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=128, hidden_size=hidden_size,
            num_layers=n_lstm_layers, batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0,
            bidirectional=False
        )
        self.dropout_lstm = nn.Dropout(dropout)

        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout_fc = nn.Dropout(0.2)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout_cnn(x)
        x = x.permute(0, 2, 1)
        lstm_out, (hidden, cell) = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.dropout_lstm(x)
        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x).squeeze()

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total trainable parameters: {total:,}')
    return total

if __name__ == '__main__':
    model = FloodCNNLSTM(n_features=21, seq_length=24)
    count_parameters(model)
    dummy = torch.randn(32, 24, 21)
    out = model(dummy)
    print(f'Output shape: {out.shape}')
    print(f'Output range: {out.min():.3f} to {out.max():.3f}')
    print('Model architecture test PASSED!')