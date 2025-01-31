import imageio
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torchvision.utils import save_image

to_pil_image = transforms.ToPILImage()

def image_to_vid(images):
    # save evolving images along the learning and get the video
    imgs = [np.array(to_pil_image(img)) for img in images]
    imageio.mimsave('outputs/generated_images.gif', imgs)

def save_reconstructed_images(recon_images, epoch, season = ''):
    # save all reconstructed images at each epoch
    save_image(recon_images.cpu(), f"outputs/image_record/{season}output{epoch}.jpg")

def save_ex(recon_ex, epoch, season = ''):
    # save an example of image at a given epoch
    save_image(recon_ex.cpu(), f"outputs/image_record/{season}ex{epoch}.jpg")

def save_loss_plot(train_loss, valid_loss, season = ''):
    # saves the plot of both losses evolutions
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'outputs/{season}loss.jpg')
    plt.show()
