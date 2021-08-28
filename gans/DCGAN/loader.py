import torchvision
import torch
img_size=64
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((img_size, img_size)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
def train_loader_fn(batch_size):
    '''
    It loads the training dataset. Takes one parameter:
    batch_size: The batch size to be used during training
    '''
    train_dataset = torchvision.datasets.CIFAR10(root="~/torch_datasets", transform=transform,download=False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader