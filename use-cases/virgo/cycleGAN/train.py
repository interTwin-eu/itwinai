import torch
import torch.nn as nn
from utils import device, LAMBDA


loss_comparison = nn.BCELoss()
loss_l1 = nn.L1Loss()


def gen_x_step(real_x, real_y, gen_x, gen_x_opt, gen_y, disc_x):
    
    gen_x_opt.zero_grad()
    
    fake_x = gen_x(real_y)
    fake_y = gen_y(real_x)
    cycled_x = gen_x(fake_y)
    cycled_y = gen_y(fake_x)
    same_x = gen_x(real_x)
    
    disc_x_output = disc_x(fake_x)
    desired_x_output = torch.ones_like(disc_x_output, device=device)
    gen_x_loss_disc = loss_comparison(disc_x_output, desired_x_output)
    gen_x_loss_cycle = loss_l1(real_x, cycled_x)
    gen_y_loss_cycle = loss_l1(real_y, cycled_y)
    total_cycle_loss = gen_x_loss_cycle + gen_y_loss_cycle
    gen_x_loss_identity = loss_l1(real_x, same_x)
    gen_x_loss_total = gen_x_loss_disc + LAMBDA * total_cycle_loss + 0.5 * LAMBDA * gen_x_loss_identity
    
    gen_x_loss_total.backward()
    gen_x_opt.step()
    
    return gen_x_loss_total, fake_x
    
    
def gen_y_step(real_x, real_y, gen_y, gen_y_opt, gen_x, disc_y):
    
    gen_y_opt.zero_grad()
    
    fake_x = gen_x(real_y)
    fake_y = gen_y(real_x)
    cycled_x = gen_x(fake_y)
    cycled_y = gen_y(fake_x)
    same_y = gen_y(real_y)
    
    disc_y_output = disc_y(fake_y)
    desired_y_output = torch.ones_like(disc_y_output, device=device)
    gen_y_loss_disc = loss_comparison(disc_y_output, desired_y_output)
    gen_x_loss_cycle = loss_l1(real_x, cycled_x)
    gen_y_loss_cycle = loss_l1(real_y, cycled_y)
    total_cycle_loss = gen_x_loss_cycle + gen_y_loss_cycle
    gen_y_loss_identity = loss_l1(real_y, same_y)
    gen_y_loss_total = gen_y_loss_disc + LAMBDA * total_cycle_loss + 0.5 * LAMBDA * gen_y_loss_identity
    
    gen_y_loss_total.backward()
    gen_y_opt.step()
    
    return gen_y_loss_total, fake_y
    

def disc_x_step(real_x, real_y, disc_x, disc_x_opt, gen_x):
    
    disc_x_opt.zero_grad()
    
    fake_x = gen_x(real_y)
    
    disc_x_pred_real = disc_x(real_x)
    desired_x_output_real = torch.ones_like(disc_x_pred_real, device=device)
    disc_loss_real = loss_comparison(disc_x_pred_real, desired_x_output_real)
    disc_x_output_fake = disc_x(fake_x.detach())
    desired_x_output_fake = torch.zeros_like(disc_x_output_fake, device=device)
    disc_loss_fake = loss_comparison(disc_x_output_fake, desired_x_output_fake)
    disc_x_loss_total = 0.5 * (disc_loss_real + disc_loss_fake)
    
    disc_x_loss_total.backward()
    disc_x_opt.step()
    
    return disc_x_loss_total
    
    
def disc_y_step(real_x, real_y, disc_y, disc_y_opt, gen_y):
    
    disc_y_opt.zero_grad()
    
    fake_y = gen_y(real_x)
    
    disc_y_pred_real = disc_y(real_y)
    desired_y_output_real = torch.ones_like(disc_y_pred_real, device=device)
    disc_loss_real = loss_comparison(disc_y_pred_real, desired_y_output_real)
    disc_y_output_fake = disc_y(fake_y.detach())
    desired_y_output_fake = torch.zeros_like(disc_y_output_fake, device=device)
    disc_loss_fake = loss_comparison(disc_y_output_fake, desired_y_output_fake)
    disc_y_loss_total = 0.5 * (disc_loss_real + disc_loss_fake)
    
    disc_y_loss_total.backward()
    disc_y_opt.step()
    
    return disc_y_loss_total
    
    
def train_step(real_x, real_y, gen_x, gen_x_opt, gen_y, gen_y_opt, disc_x, disc_x_opt, disc_y, disc_y_opt):
    
    gen_x_opt.zero_grad()
    gen_y_opt.zero_grad()
    disc_x_opt.zero_grad()
    disc_y_opt.zero_grad()
    
    fake_x = gen_x(real_y)
    fake_y = gen_y(real_x)
    cycled_x = gen_x(fake_y)
    cycled_y = gen_y(fake_x)
    same_x = gen_x(real_x)
    same_y = gen_y(real_y)
    
    disc_x_pred_fake = disc_x(fake_x.detach())
    disc_x_pred_real = disc_x(real_x)
    disc_y_pred_fake = disc_y(fake_y.detach())
    disc_y_pred_real = disc_y(real_y)
    
    desired_disc_x_output_fake = torch.zeros_like(disc_x_pred_fake, device=device)
    desired_disc_x_output_real = torch.ones_like(disc_x_pred_real, device=device)
    desired_disc_y_output_fake = torch.zeros_like(disc_y_pred_fake, device=device)
    desired_disc_y_output_real = torch.ones_like(disc_y_pred_real, device=device)
    desired_gen_x_output = torch.ones_like(disc_x_pred_fake, device=device)
    desired_gen_y_output = torch.ones_like(disc_y_pred_fake, device=device)
    
    total_cycle_loss = loss_l1(real_x, cycled_x) + loss_l1(real_y, cycled_y)
    gen_x_total_loss = loss_comparison(disc_x_pred_fake, desired_gen_x_output) + LAMBDA * total_cycle_loss + 0.5 * LAMBDA * loss_l1(real_x, same_x)
    gen_y_total_loss = loss_comparison(disc_y_pred_fake, desired_gen_y_output) + LAMBDA * total_cycle_loss + 0.5 * LAMBDA * loss_l1(real_y, same_y)
    disc_x_total_loss = 0.5 * (loss_comparison(disc_x_pred_fake, desired_disc_x_output_fake) + loss_comparison(disc_x_pred_real, desired_disc_x_output_real))
    disc_y_total_loss = 0.5 * (loss_comparison(disc_y_pred_fake, desired_disc_y_output_fake) + loss_comparison(disc_y_pred_real, desired_disc_y_output_real))
    
    gen_x_total_loss.backward(retain_graph=True)
    gen_y_total_loss.backward(retain_graph=True)
    disc_x_total_loss.backward(retain_graph=True)
    disc_y_total_loss.backward()
    
    gen_x_opt.step()
    gen_y_opt.step()
    disc_x_opt.step()
    disc_y_opt.step()
    
    return gen_x_total_loss, gen_y_total_loss, disc_x_total_loss, disc_y_total_loss, fake_x, fake_y