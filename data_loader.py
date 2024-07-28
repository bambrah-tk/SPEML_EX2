import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def get_dataloader(dataset, train_path, test_path, batch_size):
    if dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        trainset = datasets.CIFAR10(root=train_path, train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = datasets.CIFAR10(root=test_path, train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        n_classes = 10
    elif dataset == 'fashionmnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        trainset = datasets.FashionMNIST(root=train_path, train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
        testset = datasets.FashionMNIST(root=test_path, train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        n_classes = 10
    return trainloader, testloader, n_classes

def getwmloader(wm_path, batch_size, wm_lbl):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    wmset = datasets.ImageFolder(root=wm_path, transform=transform)
    wmloader = torch.utils.data.DataLoader(wmset, batch_size=batch_size, shuffle=True)
    return wmloader
