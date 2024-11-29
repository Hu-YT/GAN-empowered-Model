from mpmath.identification import transforms
from torch.utils.data import DataLoader
from wandb.integration.torch.wandb_torch import torch

from data_processing import data_process
from models import Generator, Critic, get_noise
from condition import get_one_hot_labels, combine_vectors
from gradient_penalty import get_gradient, gradient_penalty
from loss import get_gen_loss, get_crit_loss
import torch.nn as nn
import torch.optim
from scipy.io import loadmat, savemat


# initializing parameters
data_shape = (10, 5)
n_classes = 17 * 72
n_epochs = 100
z_dim = 64
batch_size = 8
lr = 0.0002
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10
crit_repeats = 5
device = 'cpu'

# data_process()
filepath = 'car1.mat'
dataset = data_process(filepath)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
# for batch in dataloader:

# initializing generator, critic and optimizers
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
crit = Critic().to(device)
crit_opt = torch.optim.Adam(crit.parameters(), lr=lr, betas=(beta_1, beta_2))

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)
gen = gen.apply(weights_init)
crit = crit.apply(weights_init)

# put it all together
cur_step = 0
generator_losses = []
critic_losses = []

for epoch in range(n_epochs):
    for real in dataloader:
        cur_batch_size = len(real)
        real = real.to(device)
        
        mean_iteration_critic_loss = 0
        for _ in range(crit_repeats):
            ### Update critic ###
            crit_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device)
            fake = gen(fake_noise)
            crit_fake_pred = crit(fake.detach())
            crit_real_pred = crit(real)

            epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
            gradient = get_gradient(crit, real, fake.detach(), epsilon)
            gp = gradient_penalty(gradient)
            crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

            mean_iteration_critic_loss += crit_loss.item() / crit_repeats

            crit_loss.backward(retain_graph=True)

            crit_opt.step()
        critic_losses += [mean_iteration_critic_loss]

        ### Update generator ###
        gen_opt.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, z_dim, device)
        fake_2 = gen(fake_noise_2)
        if (epoch + 1) % 10 == 0:
            fake_numpy = fake_2.cpu().detach().numpy()
            savemat(f'generated_data_epoch_{epoch + 1}.mat', {'fake_numpy': fake_numpy})
        crit_fake_pred = crit(fake_2)

        gen_loss = get_gen_loss(crit_fake_pred)
        gen_loss.backward()

        gen_opt.step()

        generator_losses += [gen_loss.item()]

        cur_step += 1



