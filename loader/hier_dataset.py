
import os
import numpy as np
import logging
from torchvision import transforms
import torch.utils.data as data

from utils import flist_reader, default_loader

logger = logging.getLogger('mylogger')

class ImageFilelist(data.Dataset):

    def __init__(self, flist, transform=None, target_transform=None,
                 flist_reader=flist_reader, loader=default_loader):
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader


    def __getitem__(self, index):
        iminfo = self.imlist[index]
        impath, target, index = iminfo[:3]

        img = self.loader(impath)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


    def __len__(self):
        return len(self.imlist)


def hierdata(splits, **kwargs):
    """Function to build data loader(s) for the specified splits given the parameters.
    """
    default_path = '/path/to/Deep-RTC/prepro/data/inaturalist'
    data_path = kwargs.setdefault('data_path', default_path)
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('data_path', None)

    data_loader = dict()
    for split in splits:

        if split == 'train':
            # data augmentation
            trans = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        else:
            trans = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        data_list = os.path.join(data_path, 'gt_{}.txt'.format(split))
        dataset = ImageFilelist(flist=data_list, flist_reader=flist_reader, transform=trans)

        data_loader[split] = data.DataLoader(
            dataset, shuffle=True, drop_last=False, pin_memory=True, **kwargs
        )

        print("{split}: {size}".format(split=split, size=len(dataset)))
        logger.info("{split}: {size}".format(split=split, size=len(dataset)))

    data_loader['nodes'] = np.load(os.path.join(data_path, 'leaf_nodes.npy')).tolist()
    data_loader['node_labels'] = np.load(os.path.join(data_path, 'tree.npy')).tolist()
    print("Building data loader with {} workers".format(num_workers))
    logger.info("Building data loader with {} workers".format(num_workers))

    return data_loader


def hierdata_cifar(splits, **kwargs):
    """Function to build data loader(s) for the specified splits given the parameters.
    """
    default_path = '/path/to/Deep-RTC/prepro/data/cifar100'
    data_path = kwargs.setdefault('data_path', default_path)
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('data_path', None)

    data_loader = dict()
    for split in splits:

        if split == 'train':
            # data augmentation
            trans = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        else:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

        data_list = os.path.join(data_path, 'gt_{}.txt'.format(split))
        dataset = ImageFilelist(flist=data_list, flist_reader=flist_reader, transform=trans)

        data_loader[split] = data.DataLoader(
            dataset, shuffle=True, drop_last=False, pin_memory=True, **kwargs
        )

        print("{split}: {size}".format(split=split, size=len(dataset)))
        logger.info("{split}: {size}".format(split=split, size=len(dataset)))

    data_loader['nodes'] = np.load(os.path.join(data_path, 'leaf_nodes.npy')).tolist()
    data_loader['node_labels'] = np.load(os.path.join(data_path, 'tree.npy')).tolist()
    print("Building data loader with {} workers".format(num_workers))
    logger.info("Building data loader with {} workers".format(num_workers))

    return data_loader

