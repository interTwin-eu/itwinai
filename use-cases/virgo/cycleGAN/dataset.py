import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_preprocessing import load_image_train, load_image_test

dataset_folder_name = 'aux_channel_two_cycle'

BATCH_SIZE = 1
IMG_SIZE = 256

class Dataset(Dataset):
    def __init__(self, dir_path, size=IMG_SIZE, train=True):
        self.is_train = train
        self.sizes = (size, size)
        strain_images = []
        for data in os.listdir(dir_path + 'strain/'):
            if data[-4:] == '.png':
                item = os.path.join(dir_path + 'strain/', data)
                strain_images.append(item)
        aux_images = []
        for data in os.listdir(dir_path + 'aux/'):
            if data[-4:] == '.png':
                item = os.path.join(dir_path + 'aux/', data)
                aux_images.append(item)
        self.strain = strain_images
        self.aux = aux_images
        self.length_dataset = max(len(self.strain), len(self.aux))
        self.strain_len = len(self.strain)
        self.aux_len = len(self.aux)
    def __len__(self):
        return self.length_dataset
    def __getitem__(self, idx):
        strain_img = self.strain[idx % self.strain_len]
        aux_img = self.aux[idx % self.aux_len]
        if self.is_train:
            strain_img, real_aux_img = load_image_train(strain_img)
            aux_img, real_strain_img = load_image_train(aux_img)
        else:
            strain_img, real_aux_img = load_image_test(strain_img)
            aux_img, real_strain_img = load_image_test(aux_img)
        return (strain_img, real_aux_img), (aux_img, real_strain_img)
    
data_path = f'../{dataset_folder_name}/train/'
ds = Dataset(data_path)

dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
