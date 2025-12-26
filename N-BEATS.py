import torch
import torch.nn as nn
import numpy as np
import pandas as pd

# Define a single N-BEATS block
class NBeatsBlock(nn.Module):
    def __init__(self, input_size, hidden_size, theta_size, horizon):
        super(NBeatsBlock, self).__init__()
        self.input_size = input_size
        self.horizon = horizon
        self.theta_size = theta_size

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, theta_size)
        )
        self.backcast_basis = nn.Parameter(torch.randn(theta_size, input_size))
        self.forecast_basis = nn.Parameter(torch.randn(theta_size, horizon))

    def forward(self, x):
        theta = self.fc(x)
        backcast = theta @ self.backcast_basis
        forecast = theta @ self.forecast_basis
        return backcast, forecast

# Define the complete N-BEATS model
class NBeats(nn.Module):
    def __init__(self, input_size, hidden_size, theta_size, horizon, num_blocks):
        super(NBeats, self).__init__()
        self.blocks = nn.ModuleList([
            NBeatsBlock(input_size, hidden_size, theta_size, horizon)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        residuals = x
        forecast = torch.zeros(x.size(0), self.blocks[0].horizon).to(x.device)
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast
        return forecast

# Train the model
def train_model(model, data, input_size, horizon, epochs=100, min_loss=1e-6):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    # Use the first input_size points as input and the next horizon points as the target
    X = torch.tensor(data[:input_size], dtype=torch.float32).unsqueeze(0)  # Shape: (1, input_size)
    y = torch.tensor(data[input_size:input_size + horizon], dtype=torch.float32).unsqueeze(0)  # Shape: (1, horizon)

    for epoch in range(epochs):
        model.zero_grad()
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")
        # Stop training when the loss is smaller than a predefined threshold
        if loss.item() < min_loss:
            print(f"Loss reached {loss.item()} < {min_loss}, stopping training at epoch {epoch}")
            break

    return model
