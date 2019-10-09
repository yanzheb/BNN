from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
import torch
from PIL import Image
from constants import Constant

train_dir = 'dogcat'
train_files = os.listdir(train_dir)

data_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


class CatDogDataset(Dataset):
    def __init__(self, file_list, dir_, mode='train', transform=None):
        self.file_list = file_list
        self.dir = dir_
        self.mode = mode
        self.transform = transform
        if self.mode == 'train':
            if 'dog' in self.file_list[0]:
                self.label = 1
            else:
                self.label = 0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.dir, self.file_list[idx]))
        if self.transform:
            img = self.transform(img)
        if self.mode == 'train':
            img = img.numpy()
            return img.astype('float32'), self.label
        else:
            img = img.numpy()
            return img.astype('float32'), self.file_list[idx]


def get_dataset(batch_size, test_batch_size):
    cat_files = [tf for tf in train_files if 'cat' in tf]
    dog_files = [tf for tf in train_files if 'dog' in tf]

    cats = CatDogDataset(cat_files, train_dir, transform=data_transform)
    dogs = CatDogDataset(dog_files, train_dir, transform=data_transform)

    catdogs = ConcatDataset([cats, dogs])

    torch.manual_seed(Constant.random_seed)
    full_size = len(train_files)
    train_size = int(0.9 * full_size)
    test_size = full_size - train_size
    train, test = torch.utils.data.random_split(catdogs, [train_size, test_size])
    torch.manual_seed(torch.initial_seed())
    return DataLoader(train, batch_size=batch_size, shuffle=True), DataLoader(test, batch_size=test_batch_size,
                                                                              shuffle=False)
