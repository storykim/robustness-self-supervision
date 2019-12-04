import os
import glob
import numpy as np
import torch
import torchvision.transforms as trn
import torchvision.datasets as dset


def load_cifar10c_names_and_label(path):
    names = []
    y = None

    for filename in glob.glob(os.path.join(path, '*.npy')):
        if 'labels' in filename:
            with open(filename, 'rb') as f:
                y = np.load(f)
        else:
            names.append(filename)
    assert y is not None
    return names, y


def make_inf_loop(loader):
    while True:
        for x in iter(loader): yield x


def load_cifar10c_for_validation(path, prefetch):
    y = None
    x_list = []
    for filename in glob.glob(os.path.join(path, '*.npy')):
        with open(filename, 'rb') as f:
            d = np.load(f)
            if 'labels' in filename:
                y = d
            else:
                x_list += [d[:100], d[10000:10100], d[20000:20100], d[30000:30100], d[40000:40100]]

    assert y is not None
    y = y[:100]
    y = np.tile(y, 95)
    x = np.concatenate(x_list, axis=0)
    x = np.rollaxis(x, 3, 1)
    x = x.astype(np.float32) / 255.
    tensor_y = torch.from_numpy(y).long()
    tensor_x = torch.from_numpy(x)

    meta_dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)

    inf_test_loader = make_inf_loop(torch.utils.data.DataLoader(
        meta_dataset, batch_size=25, shuffle=True,
        num_workers=prefetch, pin_memory=True))
    return inf_test_loader


def get_cifar_dataloader(dataset, data_path, batch_size, test_bs, prefetch):
    train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                   trn.ToTensor()])
    test_transform = trn.Compose([trn.ToTensor()])

    if dataset == 'cifar10':
        train_data = dset.CIFAR10(data_path, train=True, transform=train_transform)
        test_data = dset.CIFAR10(data_path, train=False, transform=test_transform)
        num_classes = 10
    else:
        train_data = dset.CIFAR100(data_path, train=True, transform=train_transform)
        test_data = dset.CIFAR100(data_path, train=False, transform=test_transform)
        num_classes = 100

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=prefetch, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=test_bs, shuffle=False,
        num_workers=prefetch, pin_memory=True)

    return train_loader, test_loader, num_classes
