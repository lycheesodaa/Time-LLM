import torch.nn as nn

from models.RevIN import RevIN


class BPNN(nn.Module):
    def __init__(self, lookback, input_channels, output_step):
        super(BPNN, self).__init__()
        self.input_dim = lookback * input_channels
        self.output_step = output_step

        self.revin = RevIN(num_features=input_channels)

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, self.output_step)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        batch_size = x_enc.shape[0]

        # Apply RevIN normalization
        x_enc = self.revin(x_enc, mode='norm')

        # Reshape from [batch_size, sequence_length, channels] to [batch_size, sequence_length * channels]
        x_flat = x_enc.reshape(batch_size, -1)

        # Forward through network
        output = self.model(x_flat)

        # Reshape to [batch_size, pred_len, 1]
        output = output.unsqueeze(-1)

        # Denormalize the output
        # First reshape to match expected format for RevIN
        output_reshaped = output.repeat(1, 1, x_enc.shape[2])

        # Apply denormalization
        output_denorm = self.revin(output_reshaped, mode='denorm')

        # Extract the first feature dimension as the final output
        # Assuming the task is univariate forecasting
        final_output = output_denorm[:, :, -1:]

        return final_output


class CNN(nn.Module):
    def __init__(self, lookback, input_channels, output_step):
        super(CNN, self).__init__()

        # Calculate the size after convolutions
        # After first conv: (lookback - kernel_size + 1) = (lookback - 3 + 1) = lookback - 2
        # After second conv: (lookback - 2 - kernel_size + 1) = (lookback - 2 - 3 + 1) = lookback - 4
        self.flatten_size = 32 * (lookback - 4)

        self.revin = RevIN(num_features=input_channels)

        self.model = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.flatten_size, 64),
            nn.ReLU(),
            nn.Linear(64, output_step)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        batch_size = x_enc.shape[0]

        # Apply RevIN normalization
        x_enc = self.revin(x_enc, mode='norm')

        # PyTorch Conv1d expects [batch_size, channels, sequence_length]
        # So we need to permute from [batch_size, sequence_length, channels]
        x_enc = x_enc.permute(0, 2, 1)

        # Forward through conv layers
        output = self.model(x_enc)

        # Reshape to [batch_size, pred_len, 1]
        output = output.unsqueeze(-1)

        # Denormalize the output
        # First reshape to match expected format for RevIN
        output_reshaped = output.repeat(1, 1, self.revin.num_features)

        # Apply denormalization
        output_denorm = self.revin(output_reshaped, mode='denorm')

        # Extract the first feature dimension as the final output
        # Assuming the task is univariate forecasting
        final_output = output_denorm[:, :, -1:]

        return final_output