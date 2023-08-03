import torch
from matplotlib import pyplot as plt


device = 'cuda'
NUM_EPOCHS = 100
OUTPUT_CHANNELS = 3
LAMBDA = 100
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
checkpoint_dir = './save_states/'


def load_checkpoint(checkpoint_file, models, optimizers, lr=lr, return_losses=False):
    directory = checkpoint_dir
    checkpoint = torch.load(directory + checkpoint_file, map_location=device)
    models[0].load_state_dict(checkpoint["gen_state"])
    models[1].load_state_dict(checkpoint["disc_state"])
    optimizers[0].load_state_dict(checkpoint["gen_opt"])
    optimizers[1].load_state_dict(checkpoint["disc_opt"])
    
    epoch = checkpoint["epoch"]
    
    print(f"=> Loading  Checkpoint from epoch {epoch}")
    
    for param_group in optimizers[0].param_groups:
        param_group["lr"] = lr
    for param_group in optimizers[1].param_groups:
        param_group["lr"] = lr
    
    if return_losses:
        return epoch, (checkpoint["loss_gen"], checkpoint["loss_disc"])
    return epoch

def save_checkpoint(models, optimizers, curr_epoch, losses, wass, mse, l1, last_checkpoint, filename):
    
    if not (last_checkpoint is None):
        directory = checkpoint_dir
        last_checkpoint = torch.load(directory + last_checkpoint)
        gen_loss = last_checkpoint["loss_gen"]
        disc_loss = last_checkpoint["loss_disc"]
        wass_error = last_checkpoint["wass_error"]
        mse_error = last_checkpoint["mse_error"]
        l1_error = last_checkpoint["l1_error"]
    else:
        gen_loss = []
        disc_loss = []
        wass_error = []
        mse_error = []
        l1_error = []
    
    gen_loss.append(losses[0].item())
    disc_loss.append(losses[1].item())
    wass_error.append(wass)
    mse_error.append(mse)
    l1_error.append(l1)
    
    print("=> Saving Checkpoint")
    checkpoint = {
        "gen_state": models[0].state_dict(),
        "gen_opt": optimizers[0].state_dict(),
        "disc_state": models[1].state_dict(),
        "disc_opt": optimizers[1].state_dict(),
        "epoch": curr_epoch,
        "loss_gen": gen_loss,
        "loss_disc": disc_loss,
        "wass_error": wass_error,
        "mse_error": mse_error,
        "l1_error": l1_error
    }
    directory = checkpoint_dir
    extension = '.pth'
    torch.save(checkpoint, directory + filename + extension)


def show_images(test_input, tar, prediction):
    fig ,axs = plt.subplots(1, 3, figsize=(7,2))

    axs[0].imshow(test_input[0].detach().permute(1, 2, 0).cpu() * 0.5 + 0.5)
    axs[0].set_title("Observed Image", fontsize=8)
    axs[1].imshow(tar[0].detach().permute(1, 2, 0).cpu() * 0.5 + 0.5)
    axs[1].set_title("Ground Truth", fontsize=8)
    axs[2].imshow(prediction[0].detach().permute(1, 2, 0).cpu() * 0.5 + 0.5)
    axs[2].set_title("Generated Output", fontsize=8)
    
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')