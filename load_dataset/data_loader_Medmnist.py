"""
Create train, valid, test iterators for LC25000 [1].
Easily extended to MNIST, CIFAR-100 and Imagenet.
[1]: https://discuss.pytorch.org/t/feedback-on-pytorch-for-kaggle-competitions/2252/4
"""

import numpy as np

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as tdata
import medmnist
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import Utils

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def get_train_valid_loader(data_name,
                           batch_size,
                           subset_size=1,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    error_msg1 = "[!] valid_size should be in the range [0, 1]."
    error_msg2 = "[!] subset_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg1
    assert ((subset_size >= 0) and (valid_size <= 1)), error_msg2

    normalize = transforms.Normalize(
        mean=[.5], std=[.5]
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # load the train dataset
    root_dir = Utils.get_dataset_name('DATASETS', 'root_dir')
    if data_name == 'PathMNIST':
        dataset = medmnist.PathMNIST(split="val", transform=transform, download=True, root=root_dir)
    elif data_name == 'OCTMNIST':
        dataset = medmnist.OCTMNIST(split="val", transform=transform, download=True,  root=root_dir)
    elif data_name == 'PneumoniaMNIST':
        dataset = medmnist.PneumoniaMNIST(split="val", transform=transform, download=True,  root=root_dir)
    elif data_name == 'ChestMNIST':
        dataset = medmnist.ChestMNIST(split="val", transform=transform, download=True,  root=root_dir)
    elif data_name == 'DermaMNIST':
        dataset = medmnist.DermaMNIST(split="val", transform=transform, download=True,  root=root_dir)
    elif data_name == 'RetinaMNIST':
        dataset = medmnist.RetinaMNIST(split="val", transform=transform, download=True,  root=root_dir)
    elif data_name == 'BreastMNIST':
        dataset = medmnist.BreastMNIST(split="val", transform=transform, download=True, root=root_dir)
    elif data_name == 'BloodMNIST':
        dataset = medmnist.BloodMNIST(split="val", transform=transform, download=True, root=root_dir)
    elif data_name == 'TissueMNIST':
        dataset = medmnist.TissueMNIST(split="val", transform=transform, download=True, root=root_dir)
    elif data_name == 'OrganAMNIST':
        dataset = medmnist.OrganAMNIST(split="val", transform=transform, download=True,  root=root_dir)
    elif data_name == 'OrganCMNIST':
        dataset = medmnist.OrganCMNIST(split="val", transform=transform, download=True,  root=root_dir)
    elif data_name == 'OrganSMNIST':
        dataset = medmnist.OrganSMNIST(split="val", transform=transform, download=True,  root=root_dir)
    elif data_name == 'OrganMNIST3D':
        dataset = medmnist.OrganMNIST3D(split="val",  download=True,  root=root_dir)
    elif data_name == 'NoduleMNIST3D':
        dataset = medmnist.NoduleMNIST3D(split="val",  download=True,  root=root_dir)
    elif data_name == 'AdrenalMNIST3D':
        dataset = medmnist.AdrenalMNIST3D(split="val",  download=True,  root=root_dir)
    elif data_name == 'FractureMNIST3D':
        dataset = medmnist.FractureMNIST3D(split="val",  download=True, root=root_dir)
    elif data_name == 'VesselMNIST3D':
        dataset = medmnist.VesselMNIST3D(split="val",  download=True,  root=root_dir)
    elif data_name == 'SynapseMNIST3D':
        dataset = medmnist.SynapseMNIST3D(split="val",  download=True,  root=root_dir)
    num_train = len(dataset)
    split_subset = int(np.floor(subset_size * num_train))
    indices_subset = list(range(split_subset))
    split_valid = int(np.floor(valid_size * split_subset))

    if shuffle:
        np.random.shuffle(indices_subset)

    train_idx, valid_idx = indices_subset[split_valid:], indices_subset[:split_valid]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = tdata.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = tdata.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return (train_loader, valid_loader)


def get_train_loader(data_name,
                     batch_size,
                     shuffle=True,
                     num_workers=4,
                     pin_memory=False):
    normalize = transforms.Normalize(
        mean=[.5], std=[.5]
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # load the train dataset
    root_dir = Utils.get_dataset_name('DATASETS', 'root_dir')
    if data_name == 'PathMNIST':
        dataset = medmnist.PathMNIST(split="train", transform=transform, download=True, root=root_dir)
    elif data_name == 'OCTMNIST':
        dataset = medmnist.OCTMNIST(split="train", transform=transform, download=True,  root=root_dir)
    elif data_name == 'PneumoniaMNIST':
        dataset = medmnist.PneumoniaMNIST(split="train", transform=transform, download=True,  root=root_dir)
    elif data_name == 'ChestMNIST':
        dataset = medmnist.ChestMNIST(split="train", transform=transform, download=True,  root=root_dir)
    elif data_name == 'DermaMNIST':
        dataset = medmnist.DermaMNIST(split="train", transform=transform, download=True,  root=root_dir)
    elif data_name == 'RetinaMNIST':
        dataset = medmnist.RetinaMNIST(split="train", transform=transform, download=True,  root=root_dir)
    elif data_name == 'BreastMNIST':
        dataset = medmnist.BreastMNIST(split="train", transform=transform, download=True, root=root_dir)
    elif data_name == 'BloodMNIST':
        dataset = medmnist.BloodMNIST(split="train", transform=transform, download=True, root=root_dir)
    elif data_name == 'TissueMNIST':
        dataset = medmnist.TissueMNIST(split="train", transform=transform, download=True, root=root_dir)
    elif data_name == 'OrganAMNIST':
        dataset = medmnist.OrganAMNIST(split="train", transform=transform, download=True,  root=root_dir)
    elif data_name == 'OrganCMNIST':
        dataset = medmnist.OrganCMNIST(split="train", transform=transform, download=True,  root=root_dir)
    elif data_name == 'OrganSMNIST':
        dataset = medmnist.OrganSMNIST(split="train", transform=transform, download=True,  root=root_dir)
    elif data_name == 'OrganMNIST3D':
        dataset = medmnist.OrganMNIST3D(split="train",  download=True,  root=root_dir)
    elif data_name == 'NoduleMNIST3D':
        dataset = medmnist.NoduleMNIST3D(split="train",  download=True,  root=root_dir)
    elif data_name == 'AdrenalMNIST3D':
        dataset = medmnist.AdrenalMNIST3D(split="train",  download=True,  root=root_dir)
    elif data_name == 'FractureMNIST3D':
        dataset = medmnist.FractureMNIST3D(split="train",  download=True, root=root_dir)
    elif data_name == 'VesselMNIST3D':
        dataset = medmnist.VesselMNIST3D(split="train", download=True,  root=root_dir)
    elif data_name == 'SynapseMNIST3D':
        dataset = medmnist.SynapseMNIST3D(split="train",  download=True,  root=root_dir)
    num_train = len(dataset)
    indices = list(range(num_train))

    if shuffle:
        np.random.shuffle(indices)

    train_idx = indices
    train_sampler = SubsetRandomSampler(train_idx)

    train_loader = tdata.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )

    return train_loader


def get_test_loader(data_name,
                    batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=False):
    normalize = transforms.Normalize(
        mean=[.5], std=[.5]
    )
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    # load the train dataset
    root_dir = Utils.get_dataset_name('DATASETS', 'root_dir')
    # 3通道
    if data_name == 'PathMNIST':
        dataset = medmnist.PathMNIST(split="test", transform=transform, download=True, root=root_dir)
    # 1通道
    elif data_name == 'OCTMNIST':
        dataset = medmnist.OCTMNIST(split="test", transform=transform, download=True,  root=root_dir)
    # 1通道
    elif data_name == 'PneumoniaMNIST':
        dataset = medmnist.PneumoniaMNIST(split="test", transform=transform, download=True,  root=root_dir)
    # 1通道
    elif data_name == 'ChestMNIST':
        dataset = medmnist.ChestMNIST(split="test", transform=transform, download=True,  root=root_dir)
    # 3通道
    elif data_name == 'DermaMNIST':
        dataset = medmnist.DermaMNIST(split="test", transform=transform, download=True,  root=root_dir)
    # 3通道
    elif data_name == 'RetinaMNIST':
        dataset = medmnist.RetinaMNIST(split="test", transform=transform, download=True,  root=root_dir)
    # 1通道
    elif data_name == 'BreastMNIST':
        dataset = medmnist.BreastMNIST(split="test", transform=transform, download=True, root=root_dir)
    # 3通道
    elif data_name == 'BloodMNIST':
        dataset = medmnist.BloodMNIST(split="test", transform=transform, download=True, root=root_dir)
    # 1通道
    elif data_name == 'TissueMNIST':
        dataset = medmnist.TissueMNIST(split="test", transform=transform, download=True, root=root_dir)
    # 1通道
    elif data_name == 'OrganAMNIST':
        dataset = medmnist.OrganAMNIST(split="test", transform=transform, download=True,  root=root_dir)
    # 1通道
    elif data_name == 'OrganCMNIST':
        dataset = medmnist.OrganCMNIST(split="test", transform=transform, download=True,  root=root_dir)
    # 1通道
    elif data_name == 'OrganSMNIST':
        dataset = medmnist.OrganSMNIST(split="test", transform=transform, download=True,  root=root_dir)
    # 1通道
    elif data_name == 'OrganMNIST3D':
        dataset = medmnist.OrganMNIST3D(split="test", download=True,  root=root_dir)
    # 1通道
    elif data_name == 'NoduleMNIST3D':
        dataset = medmnist.NoduleMNIST3D(split="test",  download=True,  root=root_dir)
    # 1通道
    elif data_name == 'AdrenalMNIST3D':
        dataset = medmnist.AdrenalMNIST3D(split="test", download=True,  root=root_dir)
    # 1通道
    elif data_name == 'FractureMNIST3D':
        dataset = medmnist.FractureMNIST3D(split="test", download=True, root=root_dir)
    # 1通道
    elif data_name == 'VesselMNIST3D':
        dataset = medmnist.VesselMNIST3D(split="test", download=True,  root=root_dir)
    # 1通道
    elif data_name == 'SynapseMNIST3D':
        dataset = medmnist.SynapseMNIST3D(split="test", download=True,  root=root_dir)
    
    print(dataset)
    test_loader = tdata.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True
    )

    return test_loader


if __name__ == '__main__':
    # PathMNIST, ChestMNIST, DermaMNIST, OCTMNIST, PneumoniaMNIST, RetinaMNIST,
    #                               BreastMNIST, BloodMNIST, TissueMNIST, OrganAMNIST, OrganCMNIST, OrganSMNIST,
    #                               OrganMNIST3D, NoduleMNIST3D, AdrenalMNIST3D, FractureMNIST3D, VesselMNIST3D, SynapseMNIST3D

    (train_loader, valid_loader) = get_train_valid_loader('OrganSMNIST', batch_size=100, subset_size=1,
                                    valid_size=0.1, shuffle=True, num_workers=4,
                                    pin_memory=True)

    # train_loader = get_train_loader('OCTMNIST', batch_size=100, shuffle=True,
    #                                 num_workers=4, pin_memory=True)
    
    # valid_loader = get_test_loader('SynapseMNIST3D', batch_size=100, shuffle=True, num_workers=4,
    #                                pin_memory=True)

    for i, (images, labels) in enumerate(train_loader):
        # print(np.shape(images))
        print(labels)
        print(np.shape(labels))
    # print(len(train_loader))
    # for i, (images, labels) in enumerate(valid_loader):
    #     print(np.shape(labels))
    # print(len(valid_loader))