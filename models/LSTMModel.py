import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.RevIN import RevIN
from utils.decomposition import EMDDecomposition, WaveletDecomposition


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, window_norm=False):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.input_size = input_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.window_norm = window_norm
        if self.window_norm:
            self.revin = RevIN(num_features=input_size)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.window_norm:
            # Normalize input sequence
            x_enc = self.revin(x_enc, mode='norm')

        h0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_size).to(x_enc.device).to(torch.bfloat16)
        c0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_size).to(x_enc.device).to(torch.bfloat16)

        out, _ = self.lstm(x_enc.to(torch.bfloat16), (h0, c0))
        out = self.fc(out[:, -1, :])

        out = out.unsqueeze(-1)  # reshape to standardised (batch_size, pred_len, 1)

        # Denormalize the output if using RevIN
        if self.window_norm:
            # Expand output to match input feature dimension for proper denormalization
            out_expanded = out.repeat(1, 1, self.input_size)
            out_denorm = self.revin(out_expanded, mode='denorm')
            # Extract only the last feature dimension (target variable)
            out = out_denorm[:, :, -1:]

        return out


class Decomp_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, decomp_type='emd', window_norm=False):
        super(Decomp_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.input_size = input_size
        num_exog_features = input_size - 1  # enc_in - 1

        self.decomp_type = decomp_type  # wavelet decomp's methods are slightly different from the rest
        if self.decomp_type == 'wavelet':
            self.decomp = WaveletDecomposition()
            self.input_size = self.decomp.level + 1 + num_exog_features
        else:
            # EMD, EEMD or CEEMDAN
            self.decomp = EMDDecomposition(decomp_type, max_imfs=9)
            self.input_size = self.decomp.max_imfs + 2 + num_exog_features

        self.lstm = nn.LSTM(self.input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

        self.window_norm = window_norm
        if self.window_norm:
            self.revin = RevIN(num_features=input_size)

    def decompose_sequence(self, x):
        device = x.device
        bsz, sq_len, _  = x.shape
        x = x.reshape(bsz, -1).cpu().numpy()  # shape is (bsz, sq_len * 1)

        output = []
        for i in range(bsz):
            batched_output = self.decomp.decompose(x[i]).T
            output.append(batched_output)

        if self.decomp_type == 'wavelet':
            return torch.tensor(np.array(output), device=device)
        else:
            padded = np.zeros((bsz, sq_len, self.decomp.max_imfs + 2))
            for i, imfs in enumerate(output):
                padded[i, :, :imfs.shape[-1]] = imfs

            return torch.tensor(padded, device=device)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.window_norm:
            # Normalize input sequence
            x_enc = self.revin(x_enc, mode='norm')

        # decompose target channel into `self.input_size` number of channels
        if x_enc.shape[-1] > 1:
            features = x_enc[:, :, :-1]  # target var is at the end, extract all other features
            decomposed = self.decompose_sequence(x_enc[:, :, -1:])
            x_enc = torch.cat([features, decomposed], dim=2)
        else:
            x_enc = self.decompose_sequence(x_enc)

        h0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_size).to(x_enc.device).to(torch.bfloat16)
        c0 = torch.zeros(self.num_layers, x_enc.size(0), self.hidden_size).to(x_enc.device).to(torch.bfloat16)

        out, _ = self.lstm(x_enc.to(torch.bfloat16), (h0, c0))
        out = self.fc(out[:, -1, :])

        out = out.unsqueeze(-1)  # reshape to standardised (batch_size, pred_len, 1)

        if self.window_norm:
            # Expand output to match original input feature dimension for proper denormalization
            out_expanded = out.repeat(1, 1, self.input_size)
            out_denorm = self.revin(out_expanded, mode='denorm')
            # Extract only the last feature dimension (target variable)
            out = out_denorm[:, :, -1:]

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

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec, mask=None):
        batch_size, seq_len, _ = x.size() # (batch_size, seq_len, input_size)

        x = x.permute(0, 2, 1) # (batch_size, input_size, seq_len)
        x = self.conv(x)

        x = x.permute(0, 2, 1) # (batch_size, seq_len, hidden_size)
        _, (h_n, _) = self.lstm(x)

        out = self.fc(h_n[-1])

        out = out.unsqueeze(-1)  # reshape to standardised (batch_size, pred_len, 1)

        return out


class LSTMGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        out, h_n = self.gru(x_enc.to(torch.bfloat16))
        out = self.fc(out[:, -1, :]) # out = self.fc(h_n[-1])

        out = out.unsqueeze(-1)  # reshape to standardised (batch_size, pred_len, 1)

        return out


class GRUAttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, attention_type='scaled_dot_product', num_heads=4):
        super(GRUAttentionModel, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        if attention_type == 'basic':
            self.attention = BasicAttention(hidden_size)
        elif attention_type == 'scaled_dot_product':
            self.attention = ScaledDotProductAttention(hidden_size)
        elif attention_type == 'multi_head':
            self.attention = MultiHeadAttention(hidden_size, num_heads)
        else:
            raise ValueError("Invalid attention type")

        self.fc = nn.Linear(hidden_size, output_size)
        self.attention_type = attention_type

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # x: [batch_size, seq_len, input_size]

        # output: [batch_size, seq_len, hidden_size]
        # hidden: [1, batch_size, hidden_size]
        output, hidden = self.gru(x_enc.to(torch.bfloat16))

        if self.attention_type == 'basic':
            # attn_weights: [batch_size, 1, seq_len]
            attn_weights = self.attention(hidden[-1], output)
            # context: [batch_size, hidden_size]
            context = attn_weights.bmm(output).squeeze(1)
        elif self.attention_type == 'scaled_dot_product':
            # context: [batch_size, hidden_size]
            context = self.attention(hidden[-1].unsqueeze(1), output, output).squeeze(1)
        elif self.attention_type == 'multi_head':
            # context: [batch_size, hidden_size]
            context = self.attention(hidden[-1].unsqueeze(1), output, output).squeeze(1)

        # output: [batch_size, output_size]
        output = self.fc(context)

        output = output.unsqueeze(-1)  # reshape to standardised (batch_size, pred_len, 1)

        return output


class BasicAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BasicAttention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]
        batch_size = encoder_outputs.size(0)
        seq_len = encoder_outputs.size(1)

        # hidden: [batch_size, seq_len, hidden_size]
        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)

        # energy: [batch_size, seq_len, hidden_size]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))

        # attention: [batch_size, seq_len]
        attention = torch.sum(self.v * energy, dim=2)

        # return: [batch_size, 1, seq_len]
        return F.softmax(attention, dim=1).unsqueeze(1)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_size):
        super(ScaledDotProductAttention, self).__init__()
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).item()

    def forward(self, query, key, value):
        # query: [batch_size, 1, hidden_size]
        # key: [batch_size, seq_len, hidden_size]
        # value: [batch_size, seq_len, hidden_size]

        # energy: [batch_size, 1, seq_len]
        energy = torch.bmm(query, key.transpose(1, 2)) / self.scale

        # attention: [batch_size, 1, seq_len]
        attention = F.softmax(energy, dim=2)

        # return: [batch_size, 1, hidden_size]
        return torch.bmm(attention, value)


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.fc_q = nn.Linear(hidden_size, hidden_size)
        self.fc_k = nn.Linear(hidden_size, hidden_size)
        self.fc_v = nn.Linear(hidden_size, hidden_size)
        self.fc_o = nn.Linear(hidden_size, hidden_size)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).item()

    def forward(self, query, key, value):
        # query: [batch_size, 1, hidden_size]
        # key: [batch_size, seq_len, hidden_size]
        # value: [batch_size, seq_len, hidden_size]
        batch_size = query.shape[0]

        # Q, K, V: [batch_size, seq_len, hidden_size]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        # Q, K, V: [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # energy: [batch_size, num_heads, 1, seq_len]
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        # attention: [batch_size, num_heads, 1, seq_len]
        attention = torch.softmax(energy, dim=-1)

        # x: [batch_size, num_heads, 1, head_dim]
        x = torch.matmul(attention, V)

        # x: [batch_size, 1, hidden_size]
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hidden_size)

        # return: [batch_size, 1, hidden_size]
        x = self.fc_o(x)
        return x
