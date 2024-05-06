import numpy as np
import torch
import torchvision
from torchvision.datasets import ImageFolder
import transforms_Erasing


def get_loader(batch_size, num_workers, use_gpu):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])



    '''
    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),  # 将图片调整为 32 x 32
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    '''

    train_transform = transforms_Erasing.Compose([ # add random erasing to transforms
        transforms_Erasing.Resize((32, 32)),  # 将图片调整为 32 x 32
        transforms_Erasing.RandomCrop(32, padding=4),
        transforms_Erasing.RandomHorizontalFlip(),
        transforms_Erasing.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
        transforms_Erasing.RandomErasing(probability = 0.5, sh = 0.4, r1 = 0.3, ),
    ])



    '''
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),  # 将图片调整为 32 x 32
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])
    '''
    test_transform = transforms_Erasing.Compose([
        transforms_Erasing.Resize((32, 32)),  # 将图片调整为 32 x 32
        transforms_Erasing.RandomCrop(32, padding=4),
        transforms_Erasing.ToTensor(),
        torchvision.transforms.Normalize(mean, std),
    ])

    ##dataset_dir = 'drive/MyDrive/pytorch_shake_shake-master/flowers'
    print("load flowers")
    train_dataset_dir = '/home1/zhienyu/ondemand/data/sys/myjobs/projects/default/2/567_shake_RE/train'
    test_dataset_dir = '/home1/zhienyu/ondemand/data/sys/myjobs/projects/default/2/567_shake_RE/test'
    
    train_dataset = ImageFolder(root=train_dataset_dir,transform=train_transform)
    test_dataset = ImageFolder(root=test_dataset_dir,transform=test_transform)
    print("load flowers test and train")
    
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    return train_loader, test_loader


