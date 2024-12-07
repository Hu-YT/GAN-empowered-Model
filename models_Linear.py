import torch
from torch import nn



class Generator(nn.Module):
    def __init__(self, input_dim, output_dim=50, hidden_dim=128):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 2),
            self.make_gen_block(hidden_dim * 2, hidden_dim * 4), #DCGAN
            self.make_gen_block(hidden_dim * 4, hidden_dim * 8),
            self.make_gen_block(hidden_dim * 8, output_dim, final_layer=True),
        )

    def make_gen_block(self, input_channels, output_channels, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
                nn.BatchNorm1d(output_channels),
                nn.ReLU(inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
                nn.Tanh(),
                # nn.Sigmoid(),
            )

    def forward(self, noise):
        #x = noise.view(len(noise), self.input_dim, 1, 1)
        # print(noise.size())
        x = self.gen(noise)
        x = x.view(-1, 10, 5)
        return x

def get_noise(n_samples, z_dim, device='cuda'):
    return torch.randn(n_samples, z_dim, device=device)

class Critic(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=128):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(input_dim, hidden_dim * 2),
            self.make_crit_block(hidden_dim * 2, hidden_dim * 4),
            self.make_crit_block(hidden_dim * 4, output_dim, final_layer=True),
        )

    def make_crit_block(self, input_channels, output_channels, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
                nn.BatchNorm1d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Linear(input_channels, output_channels),
            )

    def forward(self, data):
        data = data.view(-1, 10 * 5)
        crit_pred = self.crit(data)
        return crit_pred.view(len(crit_pred), -1)