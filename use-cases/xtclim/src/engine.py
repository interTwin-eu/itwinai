from tqdm import tqdm
import torch
from initialization import pixel_wise_criterion

def final_loss(bce_loss, mu, logvar, beta=0.1):
    """
    Adds up reconstruction loss (BCELoss) and Kullback-Leibler divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)

    Parameters:
    bce_loss: recontruction loss
    mu: the mean from the latent vector
    logvar: log variance from the latent vector
    beta: weight over the KL-Divergence
    """
    BCE = bce_loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta*KLD

def train(model, dataloader, dataset, device, optimizer, criterion, beta):
    # trains the model over shuffled data set
    """
    Trains the CVAE network and returns the loss.

    Parameters:
    model: neural network (CVAE)
    dataloader: train data shuffled and split into batches
    dataset: original data
    device: CPU or GPU
    optimizer: Adam
    criterion: loss metric
    beta: weight for the KL divergence
    """
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        data = data[0]
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(data)
        bce_loss = criterion(reconstruction, data)
        # total loss = reconstruction loss + KL divergence
        loss = final_loss(bce_loss, mu, logvar, beta)
        loss.backward() # backpropagate loss to learn from mistakes
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter # average loss over the batches
    return train_loss

def validate(model, dataloader, dataset, device, criterion, beta):
    """
    Evaluates the CVAE network and returns the loss and reconstructions.
    No backpropagation.

    Parameters:
    model: neural network (CVAE)
    dataloader: test data
    dataset: original data
    device: CPU or GPU
    criterion: loss metric
    beta: weight for the KL divergence
    """
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            data= data[0]
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            bce_loss = criterion(reconstruction, data)
            loss = final_loss(bce_loss, mu, logvar, beta)
            running_loss += loss.item()
            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images

def evaluate(model, dataloader, dataset, device,
             criterion, pixel_wise_criterion = pixel_wise_criterion):
    """
    Evaluates the CVAE network and returns the reconstruction loss
    (no KL divergence component) and reconstructions.
    No backpropagation.

    Parameters:
    model: neural network (CVAE)
    dataloader: test data
    dataset: original data
    device: CPU or GPU
    criterion: loss metric
    pixel_wise_criterion: loss metric computed and returned per pixel
    """
    model.eval()
    running_loss = 0.0
    losses = []
    counter = 0
    recon_images = []
    pixel_wise_losses = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            data= data[0]
            data = data.to(device)
            reconstruction, _, _ = model(data)
            # evaluate anomalies with reconstruction error only
            loss = criterion(reconstruction, data)
            pixel_wise_losses.append(pixel_wise_criterion(reconstruction,
                                                          data))
            running_loss += loss.item()
            losses.append(loss.item()) # keep track of all losses
            # save output of every evaluation
            recon_images.append(reconstruction)
    val_loss = running_loss / counter
    return val_loss, recon_images, losses, pixel_wise_losses

def latent_space_position(model, dataloader, dataset, device, criterion):
    """
    Evaluates the CVAE network and returns the reconstruction loss
    (no KL divergence component) and reconstructions (no backpropagation).
    Returns the means and variances of the latent encodings.

    Parameters:
    model: neural network (CVAE)
    dataloader: test data
    dataset: original data
    device: CPU or GPU
    criterion: loss metric
    """
    model.eval()
    running_loss = 0.0
    counter = 0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            data= data[0]
            data = data.to(device)
            reconstruction, mu, logvar = model(data)
            if i == 0:
                mus = mu
                logvars = logvar
            else:
                mus = torch.cat((mus, mu), 0)
                logvars = torch.cat((logvars, logvar), 0)
            loss = criterion(reconstruction, data)
            running_loss += loss.item()
            # save the last batch input and output of every epoch
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                recon_images = reconstruction
    val_loss = running_loss / counter
    return val_loss, recon_images, mus, logvars
