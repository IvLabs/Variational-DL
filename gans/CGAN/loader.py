import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                  ]))

validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
                                  ]))

#Hyper parameters
batch_size = 128
d_lr = 2e-4 #learning rate of discriminator
g_lr = 2e-4 #learning rate of generator
epochs = 20

train_shape = training_data.data.shape[0]
training_loader = DataLoader(training_data,batch_size=batch_size, shuffle=True,pin_memory=True)
validation_loader = DataLoader(validation_data,batch_size=16,shuffle=True,pin_memory=True)