import os
import torch
from torch.utils import data
import torchvision.datasets as dsets
from torchvision import transforms


def get_loader(split, transform):
    dataset = dsets.STL10(root='./data/', split=split, transform=transform, download=False)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=25, shuffle=False)
    return dataloader
