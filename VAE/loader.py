import torch
import torchvision
from torchvision import transforms,datasets,models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_set = datasets.MNIST('~/torch_datasets', train=True, download=True)
test_set = datasets.MNIST('~/torch_datasets', train=False, download=True)
train_img = train_set.data.numpy()
test_img = test_set.data.numpy()

def train_loader_fn():
    train_images = torch.Tensor(train_img).view(train_img.shape[0],1,28,28).to(device)
    return train_images
def test_loader_fn():
    test_images = torch.Tensor(test_img).view(test_img.shape[0],1,28,28).to(device)
    return test_images