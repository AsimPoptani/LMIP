from turtle import forward
import torch
import torch.nn as nn


class LSTMGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.lstm = nn.LSTMCell(input_size=500, hidden_size=500)

        self.pitch = nn.Linear(in_features=500, out_features=128)

        self.velocity = nn.Linear(in_features=500, out_features=128)

        self.duration = nn.Linear(in_features=500, out_features=1)

        self.step = nn.Linear(in_features=500, out_features=1)

        self.linear1 = nn.Linear(in_features=500, out_features=500)

        self.linear2 = nn.Linear(in_features=500, out_features=500)

    def forward(self, random_or_info):

        # random_or_info size of batch,500
        lstm = self.lstm(random_or_info)[0]

        lstm = nn.LeakyReLU()(lstm)

        lin1 = self.linear1(lstm)
        lin1 = nn.LeakyReLU()(lin1)

        info = self.linear2(lin1)

        info = nn.LeakyReLU()(info)

        pitch = self.pitch(info)

        pitch = nn.LeakyReLU()(pitch)

        velocity = self.velocity(info)

        velocity = nn.LeakyReLU()(velocity)

        duration = self.duration(info)

        duration = nn.ReLU()(duration)

        step = self.step(info)

        step = nn.ReLU()(step)

        # Return pitch (0-127), velocity (0-127) , duration (float), step (float) and info (tensor)
        return pitch, velocity, duration, step, info
