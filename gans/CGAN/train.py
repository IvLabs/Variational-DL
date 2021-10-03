import torch
import loader
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import network
device = loader.device

noise_dim = 100
initial_img_size = 2
final_img_size = 32
img_channels = 3
num_classes = 10

batch_size = loader.batch_size
epochs = loader.epochs

training_loader = loader.training_loader
train_helper_Discriminator_1 = network.train_helper_Discriminator_1
train_helper_Discriminator_2 = network.train_helper_Discriminator_2
train_helper_Generator = network.train_helper_Generator
gen_net = network.gen_net
dis_net = network.dis_net

dis_cost_list = []
gen_cost_list = []

def train(X):
    for epoch in range(0,epochs):
        for i in training_loader:
            (data,label) = i
            data = data.to(device)
            label = label.to(device)
            noise = torch.randn(data.shape[0],noise_dim,2,2).to(device)
            real_disc_loss = train_helper_Discriminator_1(data,label)
            gen_data = gen_net(noise,label)
            fake_disc_loss = train_helper_Discriminator_2(gen_data,label)
            disc_loss = real_disc_loss + fake_disc_loss
            dis_cost_list.append(disc_loss.item())
            gen_loss = train_helper_Generator(gen_data,label)
            gen_cost_list.append(gen_loss.item())
        print('for epoch ',epoch+1,',Average Generative loss is:', np.mean(gen_cost_list))
        print('for epoch ',epoch+1,',Average Discrimiantive loss is:', np.mean(dis_cost_list))


train(training_loader)

plt.plot(dis_cost_list)
plt.show()

plt.plot(gen_cost_list)
plt.show()


