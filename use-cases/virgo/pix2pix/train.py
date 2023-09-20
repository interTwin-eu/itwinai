import torch
import torch.nn as nn
from utils import device, LAMBDA

loss_comparison = nn.BCEWithLogitsLoss()
l1_loss = nn.L1Loss()


def disc_step(input_image, target, gen, disc, disc_opt):
    
    disc_opt.zero_grad()
    # First let's see how the discriminator predicts a true image
    disc_pred_r = disc(input_image, target)
    # The goal is to predict each batch as true
    goal_r = torch.ones_like(disc_pred_r, device=device)
    # How much the prediction of the discriminator deviate from this scenario?
    disc_loss_r = loss_comparison(disc_pred_r, goal_r)
    # Now, how do the discriminator predicts a fake image?
    # First let's create it
    fake_image = gen(input_image).detach()
    # Then evaluate the prediction grid
    disc_pred_f = disc(input_image, fake_image)
    # Now the goal is to predict each batch of the image as fake
    goal_f = torch.zeros_like(disc_pred_f, device=device)
    # Use the same loss as above
    disc_loss_f = loss_comparison(disc_pred_f, goal_f)
    
    # The total loss of the discriminator will be the mean of the two
    disc_loss_total = (disc_loss_f + disc_loss_r) / 2
    # Calculate the gradients for the parameters according to the loss we found
    disc_loss_total.backward()
    # Update the discriminator's parameters
    disc_opt.step()
    
    return disc_loss_total


def gen_step(input_image, target, gen, disc, gen_opt):
    
    gen_opt.zero_grad()
    fake_image = gen(input_image)
    disc_output = disc(input_image, fake_image)
    desired_output = torch.ones_like(disc_output, device=device)
    
    gan_loss = loss_comparison(disc_output, desired_output)
    regularization_term = l1_loss(fake_image, target)
    gen_loss_total = gan_loss + LAMBDA * regularization_term
    
    gen_loss_total.backward()
    gen_opt.step()
    
    return gen_loss_total, fake_image
