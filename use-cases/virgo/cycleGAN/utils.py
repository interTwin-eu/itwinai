import torch
from matplotlib import pyplot as plt


device = 'cuda'
NUM_EPOCHS = 100
OUTPUT_CHANNELS = 3
LAMBDA = 10
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
checkpoint_dir = './save_states/'


def load_checkpoint(checkpoint_file, models, optimizers, lr=lr):
    directory = checkpoint_dir
    checkpoint = torch.load(directory + checkpoint_file, map_location=device)
    models[0].load_state_dict(checkpoint["gen_x_state"])
    models[1].load_state_dict(checkpoint["gen_y_state"])
    models[2].load_state_dict(checkpoint["disc_x_state"])
    models[3].load_state_dict(checkpoint["disc_y_state"])
    optimizers[0].load_state_dict(checkpoint["gen_x_opt"])
    optimizers[1].load_state_dict(checkpoint["gen_y_opt"])
    optimizers[2].load_state_dict(checkpoint["disc_x_opt"])
    optimizers[3].load_state_dict(checkpoint["disc_y_opt"])
    
    epoch = checkpoint["epoch"]
    
    print(f"=> Loading  Checkpoint from epoch {epoch}")
    
    for param_group in optimizers[0].param_groups:
        param_group["lr"] = lr
    for param_group in optimizers[1].param_groups:
        param_group["lr"] = lr
    for param_group in optimizers[2].param_groups:
        param_group["lr"] = lr
    for param_group in optimizers[3].param_groups:
        param_group["lr"] = lr
    
    return epoch


def save_checkpoint(models, optimizers, curr_epoch, losses, mse, l1, last_checkpoint, filename):
    
    if not (last_checkpoint is None):
        directory = checkpoint_dir
        last_checkpoint = torch.load(directory + last_checkpoint)
        gen_x_loss = last_checkpoint["loss_gen_x"]
        gen_y_loss = last_checkpoint["loss_gen_y"]
        disc_x_loss = last_checkpoint["loss_disc_x"]
        disc_y_loss = last_checkpoint["loss_disc_y"]
        mse_error_x = last_checkpoint["mse_error_x"]
        mse_error_y = last_checkpoint["mse_error_y"]
        l1_error_x = last_checkpoint["l1_error_x"]
        l1_error_y = last_checkpoint["l1_error_y"]
        
    else:
        gen_x_loss = []
        gen_y_loss = []
        disc_x_loss = []
        disc_y_loss = []
        mse_error_x = []
        mse_error_y = []
        l1_error_x = []
        l1_error_y = []
        
    gen_x_loss.append(losses[0][0].item())
    gen_y_loss.append(losses[0][1].item())
    disc_x_loss.append(losses[1][0].item())
    disc_y_loss.append(losses[1][1].item())
    mse_error_x.append(mse[0])
    mse_error_y.append(mse[1])
    l1_error_x.append(l1[0])
    l1_error_y.append(l1[1])
    
    print("=> Saving Checkpoint")
    checkpoint = {
        "gen_x_state": models[0].state_dict(),
        "gen_x_opt": optimizers[0].state_dict(),
        "gen_y_state": models[1].state_dict(),
        "gen_y_opt": optimizers[1].state_dict(),
        "disc_x_state": models[2].state_dict(),
        "disc_x_opt": optimizers[2].state_dict(),
        "disc_y_state": models[3].state_dict(),
        "disc_y_opt": optimizers[3].state_dict(),
        "loss_gen_x": gen_x_loss,
        "loss_gen_y": gen_y_loss,
        "loss_disc_x": disc_x_loss,
        "loss_disc_y": disc_y_loss,
        "mse_error_x": mse_error_x,
        "mse_error_y": mse_error_y,
        "l1_error_x": l1_error_x,
        "l1_error_y": l1_error_y,
        "epoch": curr_epoch
    }
    directory = checkpoint_dir
    extension = '.pth'
    torch.save(checkpoint, directory + filename + extension)
    

def show_images(real_strain, fake_aux, real_aux, fake_strain):
    fig ,axs = plt.subplots(1, 4, figsize=(20,4))

    axs[0].imshow(real_strain[0].detach().permute(1, 2, 0).cpu() * 0.5 + 0.5)
    axs[0].set_title("Real Zebra", fontsize=8)
    axs[1].imshow(fake_aux[0].detach().permute(1, 2, 0).cpu() * 0.5 + 0.5)
    axs[1].set_title("Fake Horse", fontsize=8)
    axs[2].imshow(real_aux[0].detach().permute(1, 2, 0).cpu() * 0.5 + 0.5)
    axs[2].set_title("Real Horse", fontsize=8)
    axs[3].imshow(fake_strain[0].detach().permute(1, 2, 0).cpu() * 0.5 + 0.5)
    axs[3].set_title("Fake Zebra", fontsize=8)
    
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    axs[3].axis('off')