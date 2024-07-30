import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        h0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_size).to(x_enc.device).to(torch.bfloat16)
        c0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_size).to(x_enc.device).to(torch.bfloat16)

        out, _ = self.lstm(x_enc.to(torch.bfloat16), (h0, c0))
        out = self.fc(out[:, -1, :])

        out = out.unsqueeze(-1) # reshape to standardised (batch_size, pred_len, 1)

        return out
