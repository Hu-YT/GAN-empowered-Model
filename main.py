import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from data_processing import data_process
from models_Linear import Generator, Critic, get_noise
from gradient_penalty import get_gradient, gradient_penalty
from loss import get_gen_loss, get_crit_loss, get_c
import torch
import torch.nn as nn
import torch.optim
from scipy.io import savemat
import os


# initializing parameters
data_shape = (10, 5)
n_classes = 17 * 72
n_epochs = 2000
z_dim = 64
batch_size = 72
lr_crit = 0.00005
lr_gen = 0.00005
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10
crit_repeats = 5
device = 'cuda'

# data_process()
filepath = 'car1.mat'
dataset = data_process(filepath).unsqueeze(1)
# print(dataset.shape)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# for batch in dataloader:

# initializing generator, critic and optimizers
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.RMSprop(gen.parameters(), lr=lr_gen)
crit = Critic().to(device)
crit_opt = torch.optim.RMSprop(crit.parameters(), lr=lr_crit)

#gen_opt.load_state_dict(torch.load('gen_opt.pth'))
#crit_opt.load_state_dict(torch.load('crit_opt.pth'))

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)
    if isinstance(m, nn.BatchNorm1d):
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.constant_(m.bias, 0.0)
# gen = gen.apply(weights_init)
# crit = crit.apply(weights_init)

# put it all together
cur_step = 0
generator_losses = []
critic_losses = []
critic_real_values = []
critic_fake_values = []

# fake_noise_2 = get_noise(cur_batch_size, z_dim, device)
# fake_noise_3 = get_noise(204, z_dim, device)

for epoch in tqdm(range(n_epochs)):
    for real in dataloader:
        cur_batch_size = len(real)
        # print(cur_batch_size)
        real = real.to(device)
        # print(real.shape)
        mean_iteration_critic_loss = 0
        critic_real_value = 0
        critic_fake_value = 0
        for _ in range(crit_repeats):
            ### Update critic ###
            crit_opt.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device)
            #print(fake_noise.shape)
            fake = gen(fake_noise)
            # print(fake.shape)
            # fake = fake.view(fake.shape[0], fake.shape[1], fake.shape[2], 2, -1).mean(dim=3)
            # print(fake.shape)
            crit_fake_pred = crit(fake.detach())
            crit_real_pred = crit(real)
            C_fake = get_c(crit_fake_pred)
            C_real = get_c(crit_real_pred)
            critic_real_value += C_real / crit_repeats
            critic_fake_value += C_fake / crit_repeats
            epsilon = torch.rand(len(real), 1, 1, device=device, requires_grad=True)
            # print(real.shape, fake.shape, epsilon.shape)
            gradient = get_gradient(crit, real, fake.detach(), epsilon)
            gp = gradient_penalty(gradient)
            crit_loss = get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda)

            mean_iteration_critic_loss += crit_loss.item() / crit_repeats

            crit_loss.backward(retain_graph=True)

            crit_opt.step()
        critic_losses += [mean_iteration_critic_loss]
        critic_real_values += [critic_real_value.item()]
        critic_fake_values += [critic_fake_value.item()]
        ### Update generator ###
        gen_opt.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, z_dim, device)
        fake_2 = gen(fake_noise_2)

        # fake_2 = fake_2.view(fake_2.shape[0], fake_2.shape[1], fake_2.shape[2], 2, -1).mean(dim=3)
        #print(fake_2.shape)

        crit_fake_pred = crit(fake_2)


        gen_loss = get_gen_loss(crit_fake_pred)
        gen_loss.backward()

        gen_opt.step()

        generator_losses += [gen_loss.item()]

        if (epoch + 1) % 10 == 0:
            fake_noise_3 = get_noise(204, z_dim, device)
            fake_3 = gen(fake_noise_3)
            # fake_3 = fake_3.view(fake_3.shape[0], fake_3.shape[1], fake_3.shape[2], 2, -1).mean(dim=3)
            fake_numpy = fake_3.cpu().detach().numpy()
            # print(fake_numpy.shape)
            output_path = os.path.join(os.getcwd(), 'result', f'generated_data_epoch_{epoch + 1}.mat')
            savemat(output_path, {'fake_numpy': fake_numpy})

        cur_step += 1

# torch.save(gen_opt.state_dict(), os.path.join(os.getcwd(), 'gen_opt.pth'))

output_path = os.path.join(os.getcwd(), 'losses', f'critic_loss.mat')
savemat(output_path, {'critic_losses': critic_losses})
output_path = os.path.join(os.getcwd(), 'losses', f'generator_loss.mat')
savemat(output_path, {'generator_losses': generator_losses})
output_path = os.path.join(os.getcwd(), 'losses', f'critic_real_values.mat')
savemat(output_path, {'critic_real_values': critic_real_values})
output_path = os.path.join(os.getcwd(), 'losses', f'critic_fake_values.mat')
savemat(output_path, {'critic_fake_values': critic_fake_values})


