import os

import torchvision
from torch.utils.data import Dataset
from torchvision import transforms, datasets


class DatasetLM(Dataset):
    def __init__(self, dataset, train_val_test):
        """ Loads the data at the given path using the given index (maps tokens to indices).
        Returns a list of sentences where each is a list of token indices.
        """
        assert train_val_test in ('train', 'test', 'valid')

        self.dset = None
        self.num_classes = 10
        self.channel = 1
        self.specical_processing_list = []
        self.specical = 'normal'
        if 'FashionMNIST' in dataset:
            self.channel = 1
            normalize = transforms.Normalize(mean=[0.286], std=[0.352])
            train_trans = transforms.Compose([
                # transforms.RandomCrop(28, padding=2),  # 数据增强
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            test_trans = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
            if 'train' in train_val_test:
                self.dset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=train_trans)
            elif 'test' in train_val_test:
                self.dset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=test_trans)

        elif 'MNIST' in dataset:
            self.channel = 1
            normalize = transforms.Normalize(mean=(0.5), std=(0.5))
            train_trans = transforms.Compose([
                # transforms.Resize(224),  # resnet默认图片输入大小224*224
                transforms.ToTensor(),
                normalize
            ])
            test_trans = transforms.Compose([
                # transforms.Resize(224),
                transforms.ToTensor(),
                normalize
            ])

            if 'train' in train_val_test:
                self.dset = datasets.MNIST(root='./data', train=True, download=True, transform=train_trans)
            elif 'test' in train_val_test:
                self.dset = datasets.MNIST(root='./data', train=False, download=True, transform=test_trans)

        elif 'CIFAR10' in dataset:
            self.channel = 3
            normalize = transforms.Normalize(mean=[0.286], std=[0.352])
            train_trans = transforms.Compose([
                # transforms.RandomCrop(28, padding=2),  # 数据增强
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
            test_trans = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

            if 'train' in train_val_test:
                self.dset = datasets.CIFAR10(root='./data', train=True, download=True, transform=train_trans)
            elif 'test' in train_val_test:
                self.dset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_trans)

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        if idx in self.specical_processing_list:
            return self.specical_processing(self.dset[idx])
        return self.dset[idx]

    def add_item(self, idx):
        self.specical_processing_list.append(idx)

    def setup(self, F, type):
        self.specical_processing = F
        self.specical = type