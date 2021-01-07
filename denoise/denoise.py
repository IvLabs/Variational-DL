import matplotlib.pyplot as plt  # for plotting images
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
import os
from tqdm import tqdm  # for showing progess bars during training
import loader  # module for dataset loading

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
seed = 42  # seed is used to ensure that we get the same output every time
torch.manual_seed(seed)
batch_size = 1200  # This will give 50 batches per epoch as the train set is 60k images
epochs = 20
learning_rate = 8e-3
noise_val = 0.2  # Adjusts the noise value 
model_file = 'denoise.pth'  # Path where the model is saved/loaded 


def add_noise(inputs):
    '''
    This function is used to add random noise to the input. It takes one parameter
    inputs: PyTorch Tensor containing inputs
    '''
    noise = torch.randn_like(inputs) * noise_val
    return inputs + noise

class AE(nn.Module):
    '''
    This is the autoencoder class.
    '''
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(1,8,3,stride=1),
            nn.ReLU(),
            nn.Conv2d(8,16,5,stride=1),
            nn.ReLU(),
            nn.Conv2d(16,32,5,stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )  # The encoder network
        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(32,16,3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16,8,5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(8,1,8, stride=1),
            nn.Sigmoid(),
        )  # The decoder network

    def forward(self, features):
        x = self.encoder(features.float())
        return self.decoder(x)
    
    def encode(self, features):
        # Encodes the input to a smaller size
        return self.encoder(features.float())

    def decode(self, features):
        # Decodes the given input back to its original size
        return self.decoder(features.float())

def clrscr():  # used for clearing the screen after every epoch
    if os.name == "posix":
        # Unix/Linux/MacOS/BSD/etc
        os.system('clear')
    elif os.name in ("nt", "dos", "ce"):
        # DOS/Windows
        os.system('cls')


def train_model(model, optimizer, criterion):
    '''
    This function trains the Neural Network. Parameters are:
    model: The neural network model object,
    optimizer: The optimizer object to be used during training,
    criterion: The loss function object to be used during training
    '''
    train_loader = loader.train_loader_fn(batch_size)  # Loads the training dataset
    loss_list = []  # Stores loss after every epoch
    for epoch in tqdm(range(epochs)):  # Looping for every epoch
        loss = 0
        for batch_features, _ in tqdm(train_loader):  # Looping for every batch
            batch_features_noise = add_noise(batch_features).to(device)  # Adds noise to the batch
            optimizer.zero_grad()  # Model training starts here
            outputs = model(batch_features_noise).to(device)
            train_loss = criterion(outputs, batch_features.to(device))
            train_loss.backward()
            optimizer.step()
            loss += train_loss.item()
        loss = loss / len(train_loader)
        loss_list.append(loss)  # Stores loss in the loss_list list
        clrscr()  # clears the screen
        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))
    return loss_list

def main():
    # The main function of the code, executes automatically if run as a single file
    model = AE().to(device)  # Neural Network is declared
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()
    train_need = input("Press l to load model, t to train model, tl to load and train model: ").lower()
    # Asks user whether to load saved model or train from scratch
    if train_need == 't':
        loss_list = train_model(model, optimizer, criterion)
    elif train_need == 'l':
        model.load_state_dict(torch.load(model_file))
    elif train_need == 'tl':
        model.load_state_dict(torch.load(model_file))
        loss_list = train_model(model, optimizer, criterion)
    test_loader = loader.test_loader_fn(batch_size)  # loads the testing dataset
    test_examples = None

    with torch.no_grad():
        for batch_features in test_loader:  # Test examples are passed through the model for testing
            batch_features = batch_features[0]
            test_examples = add_noise(batch_features).to(device)
            reconstruction = model(test_examples)
            break
    try:  # Plots a graph if the training was done, else skips it
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss Value')
        plt.plot(range(len(loss_list)),loss_list)
    except:
        pass
    with torch.no_grad():
        number = 10
        plt.figure(figsize=(20, 4))
        # Below code plots 10 input images and their output
        for index in range(number):
            ax = plt.subplot(2, number, index + 1)
            plt.imshow(test_examples[index].cpu().numpy().reshape(28, 28))
            # plotting the ith test image in subplot
            plt.gray()  # changing color code to grayscale
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)  # removing axes
            ax = plt.subplot(2, number, index + 1 + number)
            plt.imshow(reconstruction[index].cpu().numpy().reshape(28, 28))
            # plotting the ith test image in subplot
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()
    if train_need == 't' or train_need == 'tl':
        # If the model was trained, it asks whether or not to save the model
        save_status=input("Enter s to save the model: ").lower()
        if save_status=='s':
            torch.save(model.state_dict(),model_file)
    
if __name__ == "__main__":
    main()