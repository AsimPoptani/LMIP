import torch
import torch.nn as nn


class LSTMDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.lstm = nn.LSTMCell(input_size=500, hidden_size=500)

        self.pitch = nn.Linear(in_features=128, out_features=500)

        self.velocity = nn.Linear(in_features=128, out_features=500)

        self.duration = nn.Linear(in_features=1, out_features=500)

        self.step = nn.Linear(in_features=1, out_features=500)

        self.linear1 = nn.Linear(in_features=500, out_features=500)

        self.linear2 = nn.Linear(in_features=500, out_features=500)

        self.linear_intermediate = nn.Linear(in_features=500, out_features=500)

        self.linear3 = nn.Linear(in_features=500, out_features=1)

    def forward(self, pitches, velocities, durations, steps):

        info = torch.zeros(500)

        for index in range(len(pitches)):
            # random_or_info size of batch,500
            lstm = self.lstm(info)[0]

            lstm = nn.LeakyReLU()(lstm)

            pitch = self.pitch(pitches[index])
            step = self.step(steps[index])
            velocity = self.velocity(velocities[index])
            duration = self.duration(durations[index])

            add = pitch + step + velocity + duration

            lin1 = self.linear1(add)

            lin1 = nn.LeakyReLU()(lin1)

            lin2 = self.linear2(lin1)

            lin2 = nn.LeakyReLU()(lin2)

            info = self.linear_intermediate(lin2)

        print("info", info.shape)
        discriminator = self.linear3(info)

        discriminator = nn.Sigmoid()(discriminator)

        # Return discriminator (tensor)
        return discriminator
