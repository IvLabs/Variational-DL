import torch
import loader
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
device = loader.device
import network
import torchvision

'''
If you want to train and test,simply run test file. If your code is trained and you just want to
Test the trained network, then remove the "import train" line
'''
# import train  


noise_dim = 100
initial_img_size = 2
final_img_size = 32
img_channels = 3
num_classes = 10

noise = torch.randn(16,100,2,2).to(device)

for data_1 in loader.validation_loader:
    data_1,lab_1 = data_1
    lab_1 = lab_1.to(device)
    break

def display(X):
    X = X.numpy()
    fig = plt.imshow(np.transpose(X, (1,2,0)), interpolation='none')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()


lab = network.gen_net.fc(lab_1)
lab = lab.view(lab.shape[0],1,initial_img_size,initial_img_size)
noise = torch.cat((noise,lab),dim = 1)

output = network.gen_net.layer(noise)

(test_img,Y) = next(iter(loader.validation_loader))

# original images from the dataset
display(torchvision.utils.make_grid(test_img.cpu(),normalize=True),)

#Generated Images 
display(torchvision.utils.make_grid(output.cpu().data,normalize = True),)