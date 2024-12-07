import torch
from torch import nn



class Generator(nn.Module):
    def __init__(self, input_dim, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        self.input_dim = input_dim
        # Build the neural network
        self.gen = nn.Sequential(
            self.make_gen_block(input_dim, hidden_dim * 2),
            self.make_gen_block(hidden_dim * 2, hidden_dim * 4), #DCGAN
            self.make_gen_block(hidden_dim * 4, hidden_dim * 8),
            self.make_gen_block(hidden_dim * 8, im_chan, final_layer=True),
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
                # nn.Tanh(),
                nn.Sigmoid(),
            )

    def forward(self, noise):
        x = noise.view(len(noise), self.input_dim, 1, 1)
        return self.gen(x)

def get_noise(n_samples, z_dim, device='cuda'):
    return torch.randn(n_samples, z_dim, 1, 1, device=device)

class Critic(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=64):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim * 2),
            self.make_crit_block(hidden_dim * 2, hidden_dim * 4),
            self.make_crit_block(hidden_dim * 4, 1, final_layer=True),
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
        crit_pred = self.crit(data)
        return crit_pred.view(len(crit_pred), -1)