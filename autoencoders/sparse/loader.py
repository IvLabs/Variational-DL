import torchvision
import torch

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
def train_loader_fn(batch_size):
    '''
    It loads the training dataset. Takes one parameter:
    batch_size: The batch size to be used during training
    '''
    train_dataset = torchvision.datasets.MNIST(root="~/torch_datasets", train=True, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader

def test_loader_fn(batch_size):
    '''
    It loads the testing dataset. Takes one parameter:
    batch_size: The batch size to be passed to the loader
    '''
    test_dataset = torchvision.datasets.MNIST(
        root="~/torch_datasets", train=False, transform=transform, download=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return test_loader