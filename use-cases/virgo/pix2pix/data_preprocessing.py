from PIL import Image
import torch
from torchvision import transforms


IMG_HEIGHT = 256
IMG_WIDTH = 256

def load(image_path):
    # transform = transforms.PILToTensor()
    transform = transforms.ToTensor()
    
    image_file = Image.open(image_path)
    tensor_image = transform(image_file)
    
    full_width = tensor_image.shape[2]
    image_width = tensor_image.shape[2] // 2
    
    observed_image = tensor_image[:, :, image_width:].type(torch.float32)
    ground_truth = tensor_image[:, :, :image_width].type(torch.float32)
    
    return observed_image, ground_truth


# def resize(observed_image, ground_truth, size):
#     transform = transforms.transforms.Resize(286, interpolation=transforms.InterpolationMode.NEAREST)
#     new_obs = transform(observed_image)
#     new_truth = transform(ground_truth)
    
#     return new_obs, new_truth


# def random_crop(observed_image, ground_truth):
#     stacked_images = torch.stack([observed_image, ground_truth])
#     transform = transforms.RandomCrop((IMG_HEIGHT, IMG_WIDTH))
#     cropped_images = transform(stacked_images)
    
#     return cropped_images[0], cropped_images[1]


def normalize(observed_image, ground_truth):
    # observed_image = (observed_image / 127.5) - 1
    # ground_truth = (ground_truth / 127.5) - 1
    
    observed_image = observed_image * 2 - 1
    ground_truth = ground_truth * 2 - 1
    
    return observed_image, ground_truth


# def random_jitter(observed_image, ground_truth):
#     # Resize to 286x286
#     resized_obs, resized_truth = resize(observed_image, ground_truth, 286)
#     # Random crop back to 256x256
#     cropped_obs, cropped_truth = random_crop(resized_obs, resized_truth)
#     # Random mirroring
#     if torch.rand(1).item() > 0.5:
#         mirrored_obs = cropped_obs.flip(2)
#         mirrored_truth = cropped_truth.flip(2)
#         return mirrored_obs, mirrored_truth
        
#     return cropped_obs, cropped_truth


def load_image_train(image_path):
    observed_image, ground_truth = load(image_path)
    # observed_image, ground_truth = random_jitter(observed_image, ground_truth)
    observed_image, ground_truth = normalize(observed_image, ground_truth)

    return observed_image, ground_truth


def load_image_test(image_path):
    observed_image, ground_truth = load(image_path)
    observed_image, ground_truth = normalize(observed_image, ground_truth)

    return observed_image, ground_truth