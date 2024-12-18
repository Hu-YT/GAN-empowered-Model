import torch

def get_gen_loss(crit_fake_pred):
    gen_loss = -1. * torch.mean(crit_fake_pred)
    # gen_loss = torch.mean(crit_fake_pred)
    # print('gen_loss:', torch.mean(crit_fake_pred))
    return gen_loss

def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    crit_loss = torch.mean(crit_fake_pred) - torch.mean(crit_real_pred) + c_lambda * gp
    # print('C(x)(critic_fake_pred):', torch.mean(crit_fake_pred))
    # print('C(G(z))(critic_real_pred):', torch.mean(crit_real_pred))
    return crit_loss
def get_c(pred):
    c_value = torch.mean(pred)
    #print('C_fake:', c_fake)
    return c_value


'''
 Analysis
LG = - E(C(fake))

 '''

