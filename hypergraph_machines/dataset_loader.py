import torch
import torchvision
from torchvision import datasets, transforms

def load_dataset(name, batch_size, data_folder = "./data"):
    if name == "FashionMNIST":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
        train_loader = torch.utils.data.DataLoader(
                            datasets.FashionMNIST(data_folder, train=True,
                                                  download=True,
                                                  transform=transform),
                            drop_last=True, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
                            datasets.FashionMNIST(data_folder, train=False,
                                                  download=True,
                                                  transform=transform),
                            drop_last=True, batch_size=batch_size, shuffle=True)
        image_size = (1,28,28)
    elif name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])
        train_loader = torch.utils.data.DataLoader(
                            datasets.MNIST(data_folder, train=True,
                                           download=True,
                                           transform=transform),
                            drop_last=True, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(
                           datasets.MNIST(data_folder, train=False,
                                          download=True,
                                          transform=transform),
                           drop_last=True, batch_size=batch_size, shuffle=True)
        image_size = (1,28,28)
    elif name == "CIFAR10":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5),
                                                             (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root=data_folder, train=True,
                                                download=True,
                                                transform=transform)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                                   shuffle=True, drop_last=True)

        testset = torchvision.datasets.CIFAR10(root=data_folder, train=False,
                                               download=True,
                                               transform=transform)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                                  shuffle=False, drop_last=True)

        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        image_size = (3,32,32)
    else:
        raise NotImplementedError("The dataset is not available")
    return train_loader, test_loader, image_size
