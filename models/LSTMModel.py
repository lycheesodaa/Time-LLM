import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        h0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_size).to(x_enc.device).to(torch.bfloat16)
        c0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_size).to(x_enc.device).to(torch.bfloat16)

        out, _ = self.lstm(x_enc.to(torch.bfloat16), (h0, c0))
        out = self.fc(out[:, -1, :])

        out = out.unsqueeze(-1)  # reshape to standardised (batch_size, pred_len, 1)

        return out


'''
The errors may be propagated extensively since we would be using predicted values for other features as well, to 
recursively predict the next time step.

Consider using teacher-forcing during training, where the actual values for the other features of future timesteps
are used instead of the predictions. 
- We would have to justify where we can obtain these 'actual values' from, since technically these would be forecast values from external sources as well.
'''
class LSTMRecursiveModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMRecursiveModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # Fencepost
        _, (h_n, c_n) = self.lstm(x_enc)

        predictions = torch.zeros(x_enc.size(0), x_enc.size(1), self.input_size, device=x_enc.device)

        rolling_window = x_enc.clone()

        for i in range(self.output_size):
            out, (h_n, c_n) = self.lstm(rolling_window, (h_n, c_n))

            current_pred = self.fc(out[:, -1, :])  # Shape: (batch_size, output_size)
            predictions[:, i, :] = current_pred

            rolling_window = torch.cat([rolling_window[:, 1:, :], current_pred.unsqueeze(1)], dim=1)

        return predictions


class ConvLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, kernel_size=3):
        super(ConvLSTMModel, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size, padding=kernel_size // 2)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = x.size()

        # Reshape for Conv1d: (batch_size, input_size, seq_len)
        x = x.permute(0, 2, 1)

        # Apply Conv1d
        x = self.conv(x)

        # Reshape back for LSTM: (batch_size, seq_len, hidden_size)
        x = x.permute(0, 2, 1)

        # LSTM
        _, (h_n, _) = self.lstm(x)

        # Use the final hidden state
        out = self.fc(h_n[-1])
        return out
