import torch

def get_gen_loss(crit_fake_pred):
    gen_loss = -1. * torch.mean(crit_fake_pred)
    # print('gen_loss:', torch.mean(crit_fake_pred))
    return gen_loss

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
    # print('C(x)(critic_fake_pred):', torch.mean(crit_fake_pred))
    # print('C(G(z))(critic_real_pred):', torch.mean(crit_real_pred))
    return crit_loss
def get_c_fake(crit_fake_pred):
    c_fake = torch.mean(crit_fake_pred)
    #print('C_fake:', c_fake)
    return c_fake
def get_c_real(crit_real_pred):
    c_real = torch.mean(crit_real_pred)
    #print('C_real', c_real)
    return c_real

'''
 Analysis
LG = - E(C(fake))

 '''

