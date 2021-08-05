import loader  #loads the MNIST dataset
import torch
import torchvision
from torchvision import transforms,datasets,models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from tqdm import tqdm
torch.seed = 41
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 400
epochs = 10
alpha = 0.005
model_file='VAE.pth'

# creates a convolution layer with 3x3 kernel size
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=stride, padding=1).to(device)

#creates a transpose convolution layer with 3x3 kernel size
def convT3x3(in_channels, out_channels, stride=1,padding=1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3,stride=stride, padding=padding).to(device)

# We are using a ResNet to train the data
# creating the residual blocks of the resnet model
# this is used as encoder residual block of the model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.subblock_1=nn.Sequential(
            conv3x3(in_channels, out_channels, stride).to(device),
            nn.BatchNorm2d(out_channels).to(device),
            nn.ReLU()
        )
        self.subblock_2=nn.Sequential(
            conv3x3(out_channels, out_channels).to(device),
            nn.BatchNorm2d(out_channels).to(device),
            nn.ReLU(),
        )
        self.downsample = downsample
    def forward(self, x):
        residual = x
        x = self.subblock_1(x).to(device)
        x = self.subblock_2(x).to(device)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = F.relu(x)        
        return x

block=ResidualBlock

#decoder residual block of the model
class DecoderResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,padding=1,upsample=None):
        super(DecoderResidualBlock, self).__init__()
        self.subblock_1=nn.Sequential(
            convT3x3(in_channels, out_channels, stride,padding).to(device),
            nn.BatchNorm2d(out_channels).to(device),
            nn.ReLU()
        )
        self.subblock_2=nn.Sequential(
            convT3x3(out_channels, out_channels).to(device),
            nn.BatchNorm2d(out_channels).to(device),
            nn.ReLU(),
        )
        self.upsample = upsample
    def forward(self, x):
        residual = x
        x = self.subblock_1(x).to(device)
        x = self.subblock_2(x).to(device)
        if self.upsample:
            residual = self.upsample(residual)
        x += residual
        x = F.relu(x)        
        return x
decoder_block=DecoderResidualBlock

# Creating the ResNet and inverse ResNet layers
class ResNet(nn.Module):
    def __init__(self,num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels=16
        self.in_channels_decoder=32
        self.layer=nn.Sequential(
            conv3x3(1, 16).to(device),
            nn.BatchNorm2d(16).to(device),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(block,16,2).to(device)
        self.layer2 = self.make_layer(block, 32,2,2).to(device)
        self.layer3 = self.make_layer(block, 64,2,2).to(device)
        self.max_pool = nn.MaxPool2d(7,1).to(device)
        self.fc = nn.Flatten()
        self.trans_layer1 = self.make_trans_layer(decoder_block,32,2).to(device)
        self.trans_layer2 = self.make_trans_layer(decoder_block, 16,2,2).to(device)
        self.trans_layer3 = self.make_trans_layer(decoder_block,3,2,2).to(device)
        self.trans_layer4 = nn.ConvTranspose2d(3,1,4,2,2).to(device)

    #making resnet layers
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride).to(device),
                nn.BatchNorm2d(out_channels).to(device)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    # making inverse resnet layer
    def make_trans_layer(self, decoder_block, out_channels, blocks, stride=1,padding=0):
        upsample = None
        if (stride != 1) or (self.in_channels_decoder != out_channels):
            upsample = nn.Sequential(
                convT3x3(self.in_channels_decoder, out_channels, stride=stride,padding=padding).to(device),
                nn.BatchNorm2d(out_channels).to(device)
            )
        layers_dec = []
        layers_dec.append(decoder_block(self.in_channels_decoder, out_channels, stride,padding, upsample))
        self.in_channels_decoder = out_channels
        for i in range(1, blocks):
            layers_dec.append(decoder_block(out_channels, out_channels))
        return nn.Sequential(*layers_dec)

    def forward(self, x):
        # encoder part
        x = self.layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.max_pool(x)

        # latent vector generation
        latent = self.fc(x)
        mean,std=torch.chunk(latent,2,dim=1)

        #sampling of latent vecotr
        sample = mean + torch.randn_like(std)*std
        x=sample.view(sample.shape[0],32,1,1)
        x = self.trans_layer1(x)
        x = self.trans_layer2(x)
        x = self.trans_layer3(x)
        x = self.trans_layer4(x)
        return x,mean,std
    def decode(self, mean, std):
        sample = mean + torch.randn_like(std)*std
        x=sample.view(sample.shape[0],32,1,1)
        x = self.trans_layer1(x)
        x = self.trans_layer2(x)
        x = self.trans_layer3(x)
        x = self.trans_layer4(x)
        return x
AutoEncoder=ResNet()

#optimizer function
optimizer = torch.optim.Adam(AutoEncoder.parameters(), lr=alpha)


# calculation of variational loss
def variational_loss(output,X_in,mean,std):
    loss_function = nn.MSELoss()
    loss_by_function = loss_function(output,X_in)
    kl_loss = -(0.5/batch_size)*torch.sum(1+torch.log(torch.pow(std,2)+1e-10)-torch.pow(std,2)-torch.pow(mean,2))
    total_loss = loss_by_function+kl_loss
    return total_loss

#training function
def train(X):
    loss_list = []
    im_list = []
    iters=0
    j=0
    for epoch in tqdm(range(0,epochs)):
        cost = 0
        batch=torch.randperm(X.shape[0]).to(device)
        for i in tqdm(range(0, X.shape[0],batch_size)):
            output,mean,std = AutoEncoder(X[batch[i:i+batch_size]].to(device))
            optimizer.zero_grad()
            loss = variational_loss(output,X[batch[i:i+batch_size]],mean,std)
            cost = cost+loss.item() 
            loss.backward()
            optimizer.step()
            
            # to generate random image 
            if (iters % 50 == 0) or ((epoch == epochs-1) and (j == len(X)-1)):
                with torch.no_grad():
                    test = AutoEncoder.decode(mean,std).detach().cpu()
                im_list.append(np.squeeze(test[0].permute(1,2,0)))
            iters+=1
            j+=1
        loss_avg = cost / X.shape[0]
        loss_list.append(loss_avg)
        print("For iteration: ", epoch+1, " the loss is :", loss_avg)
    return loss_list,im_list


def test(X):
    with torch.no_grad():
        cost = 0
        batch = torch.randperm(X.shape[0])
        for i in tqdm(range(0, X.shape[0],batch_size)):
            output,mean,std = AutoEncoder(X[batch[i:i+batch_size]])
            loss = variational_loss(output,X[batch[i:i+batch_size]],mean,std)
            cost = cost+loss.item()
        print("Test set loss:",cost/X.shape[0])

def main():
    train_need = input("Press l to load model, t to train model, tl to load and train model: ").lower()
    # Asks user whether to load saved model or train from scratch, or train the saved loss
    if train_need == 't':
        #loading train set images as tensors
        train_images = loader.train_loader_fn()
        loss_list,im_list = train(train_images)
    elif train_need == 'l':
        AutoEncoder.load_state_dict(torch.load(model_file))
    elif train_need == 'tl':
        AutoEncoder.load_state_dict(torch.load(model_file))
        #loading train set images as tensors
        train_images = loader.train_loader_fn()
        loss_list,im_list = train(train_images)
    #to save randomly generated images
    i=0
    try: 
        for l in im_list:
            i+=1
            plt.savefig(str(i))
            plt.imshow(l,cmap="gray")
        # plotting the cost function
        plt.plot(loss_list)
        plt.title("Loss curve")
        plt.ylabel('cost')
        plt.xlabel('epoch number')
        plt.show()
    except:
        pass

    # loading the test set of images
    test_images=loader.test_loader_fn()
    test(test_images)

    n = 10 # number of images that are to be displayed

    # n test images passed through variational autoencoder
    output = AutoEncoder(test_images[:n])
    output_img = ((output[0].to(torch.device('cpu'))).detach().numpy()).reshape(n,28,28)

    for i in range(0,n):
        axes = plt.subplot(2,n,i+1)
        plt.imshow(loader.test_img[i],cmap = "gray")
        axes.get_xaxis().set_visible(False) #removing axes
        axes.get_yaxis().set_visible(False)

        axes = plt.subplot(2,n,n+i+1)
        plt.imshow(output_img[i],cmap="gray")
        axes.get_xaxis().set_visible(False)
        axes.get_yaxis().set_visible(False)
    plt.show()
    if train_need == 't' or train_need == 'tl':
        # If the model was trained, it asks whether or not to save the model
        save_status=input("Enter s to save the model: ").lower()
        if save_status=='s':
            torch.save(AutoEncoder.state_dict(),model_file)

if __name__ == "__main__":
    main()


    



