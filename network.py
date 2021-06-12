import torch
import numpy as np

class Block(torch.nn.Module):
    def __init__(
            self,
            input_channel,
            output_channel,
            resolution,
            architecture='resnet'
    ):
        super(Block, self).__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.resolution = resolution
        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=2, bias=False)
        self.relu1 = torch.nn.ReLU()
        self.bn = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel, output_channel, kernel_size=3, stride=2, bias=False)
        self.relu2 = torch.nn.ReLU()
        self.architecture = architecture
    def forward(self, x):
        if self.architecture == 'resnet':
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
        else:
            x = self.conv1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.relu2(x)
        return x
class Network(torch.nn.Module):
    def __init__(
            self,
            input_channel,
            action_size,
            resolution,
            channel_max=512,
    ):
        super(Network, self).__init__()
        self.input_channel = input_channel
        self.action_size = action_size
        self.resolution = resolution
        self.resolution_log2 = int(np.log2(resolution // 16) + 1)
        self.block_resolutions = [16 * 2 ** (i - 1) for i in range(self.resolution_log2, 0, -1)]
        channels_dict = {res: min(2 ** (10 - np.log2(res // 16)), channel_max) for res in self.block_resolutions + [16]}
        for res in self.block_resolutions:
            in_channels = int(channels_dict[res]) if res < resolution else 1
            out_channels = int(channels_dict[res // 2]) if res != 16 else 512
            block = Block(in_channels, out_channels, res)
            setattr(self, f'b{res}', block)
        self.fc = torch.nn.Linear(8 * 6 * 512, action_size * 2)
        self.state_fc = torch.nn.Sequential(
                    torch.nn.Linear(action_size, action_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(action_size, 1),
                )
        self.advantage_fc = torch.nn.Sequential(
                    torch.nn.Linear(action_size, action_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(action_size, action_size)
                )
    def forward(self, x):
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            x = block(x)
        x = self.fc(x.flatten(1))
        x1 = x[:, :self.action_size]  # input for the net to calculate the state value
        x2 = x[:, self.action_size:]  # relative advantage of actions in the state
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        x = state_value + (advantage_values - advantage_values.mean(dim=1).reshape(-1, 1))
        return x

class DuelQNet(torch.nn.Module):
    """
    This is Duel DQN architecture.
    see https://arxiv.org/abs/1511.06581 for more information.
    """

    def __init__(self, available_actions_count):
        super(DuelQNet, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 32, kernel_size=8, stride=4, bias=True),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=True),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )

        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=True),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )

        self.state_fc = torch.nn.Sequential(
            torch.nn.Linear(6144, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 1)
        )

        self.advantage_fc = torch.nn.Sequential(
            torch.nn.Linear(6144, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, available_actions_count)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.flatten(1)
        x1 = x  # input for the net to calculate the state value
        x2 = x  # relative advantage of actions in the state
        state_value = self.state_fc(x1).reshape(-1, 1)
        advantage_values = self.advantage_fc(x2)
        x = state_value + (advantage_values - advantage_values.mean(dim=1).reshape(-1, 1))

        return x
