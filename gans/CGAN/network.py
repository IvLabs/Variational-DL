import torch
import loader
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
device = loader.device

noise_dim = 100
initial_img_size = 2
final_img_size = 32
img_channels = 3
num_classes = 10

d_lr = loader.d_lr
g_lr = loader.g_lr

class Generator(nn.Module):
    def __init__(self,noise_dim,img_channels,initial_img_size,num_classes):
        super().__init__()
        self.noise_dim = noise_dim
        self.img_channels = img_channels
        self.initial_img_size = initial_img_size
        self.num_classes = num_classes
        self.layer = nn.Sequential(
                        nn.ConvTranspose2d(self.noise_dim+1,512,4,2,1,bias = False), #4
                        nn.BatchNorm2d(512),
                        nn.ReLU(),
            
                        nn.ConvTranspose2d(512,256,4,2,1,bias = False), #8
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
            
                        nn.ConvTranspose2d(256,128,5,1,bias = False), #12
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
            
                        nn.ConvTranspose2d(128,64,7,1,1,bias = False), #16
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
            
                        nn.ConvTranspose2d(64,32,5,2,2,bias = False), #31
                        nn.BatchNorm2d(32),
                        nn.ReLU(),
            
                        nn.ConvTranspose2d(32,self.img_channels,4,1,1,bias = False), #32
                        nn.Tanh()
        )

        '''
        nn.Embedding layer is simply a linear layer. Coming to the below line, it considers one hot encoding of the corresponding
        label(length of vector = number of classes) and picks up the index of non-zero number in the vector and performs linear operation.
        To be more precise, it works same as that of fc layer.
        '''
        self.fc = nn.Embedding(self.num_classes,self.initial_img_size*self.initial_img_size)
    
    def forward(self,x,y):
        y = self.fc(y)
        y = y.view(y.shape[0],1,self.initial_img_size,self.initial_img_size)
        x = torch.cat((x,y),dim = 1)
        x = self.layer(x)
        return x
gen_net = Generator(noise_dim,img_channels,initial_img_size,num_classes)

class Discriminator(nn.Module):
    def __init__(self,noise_dim,img_channels,final_img_size,num_classes):
        super().__init__()
        self.noise_dim = noise_dim
        self.img_channels = img_channels
        self.final_img_size = final_img_size
        self.num_classes = num_classes
        self.dc1 = nn.Sequential(
                    nn.Conv2d(self.img_channels+1,64,4,2,1,bias = False), #16
                    nn.BatchNorm2d(64),
                    nn.LeakyReLU(0.2, inplace=True)
        )
        self.dc2 = nn.Sequential(
                    nn.Conv2d(64,128,4,2,1,bias = False), #8
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True)
        )
        self.dc3 = nn.Sequential(
                    nn.Conv2d(128,256,4,2,1,bias = False), #4
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True)
        )
        self.dc4 = nn.Sequential(
                    nn.Conv2d(256,512,4,2,1,bias = False), #2
                    nn.Sigmoid()
        )
        self.fc = nn.Embedding(self.num_classes,self.final_img_size*self.final_img_size)
    def forward(self,x,y):
        y = self.fc(y)
        y = y.view(y.shape[0],1,self.final_img_size,self.final_img_size)
        x = torch.cat((x,y),dim = 1)
        x = self.dc1(x)
        x = self.dc2(x)
        x = self.dc3(x)
        x = self.dc4(x)
        return x
dis_net = Discriminator(noise_dim,img_channels,final_img_size,num_classes)

gen_net = gen_net.to(device)
dis_net = dis_net.to(device)

def loss_function(X,Y):
    loss = nn.BCELoss()
    loss = loss(X,Y)
    return loss

dis_optimiser = torch.optim.Adam(dis_net.parameters(),lr=d_lr,betas=(0.5, 0.999))
gen_optimiser = torch.optim.Adam(gen_net.parameters(),lr=g_lr,betas=(0.5, 0.999))

def train_helper_Discriminator_1(train_img,train_label):
    dis_net.zero_grad()
    true_output = dis_net(train_img,train_label)
    real = loss_function(true_output,torch.ones_like(true_output).to(device))
    real.backward()
    return real
def train_helper_Discriminator_2(gen_img,train_label):
    fake_output = dis_net(gen_img.detach(),train_label)
    fake = loss_function(fake_output,torch.zeros_like(fake_output).to(device))
    fake.backward()
    dis_optimiser.step()
    return fake
def train_helper_Generator(gen_img,train_label):
    gen_net.zero_grad()
    final_output = dis_net(gen_img,train_label)
    gen_loss = loss_function(final_output,torch.ones_like(final_output).to(device))
    gen_loss.backward()
    gen_optimiser.step()
    return gen_loss
