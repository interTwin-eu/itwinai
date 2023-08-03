import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from data_preprocessing import load_image_train, load_image_test

data_folder_name = 'aux_channel_two'

BATCH_SIZE = 1

class Dataset(Dataset):
    def __init__(self, dir_path, size=256, train=True):
        self.is_train = train
        self.sizes = (size, size)
        items = []
        labels = []
        for data in os.listdir(dir_path):
            item = os.path.join(dir_path, data)
            items.append(item)
            labels.append(data)
        self.items = [item for item in items if item[-4:] == '.png']
        self.labels = labels
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        if self.is_train:
            data = load_image_train(self.items[idx])
        else:
            data = load_image_test(self.items[idx])
        return data, self.labels[idx]
    
data_path = f'../{data_folder_name}/train/'
ds = Dataset(data_path, size=256)

dataloader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)