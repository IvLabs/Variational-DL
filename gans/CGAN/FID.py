import network
import torch
import loader
from torchvision import transforms
from torch.utils.data import DataLoader
device = loader.device

initial_img_size = network.initial_img_size
final_img_size = network.final_img_size
num_of_imgs = 3000 # number of images that should be stored in a folder. FID is calculated on those images
noise = torch.randn(num_of_imgs,100,2,2).to(device)

validation_data = loader.validation_data
validation_loader = DataLoader(validation_data,batch_size = num_of_imgs,shuffle=True,pin_memory=True)
transPIL = transforms.ToPILImage('RGB')

for data in validation_loader:
    data,label = data
    label = label.to(device)
    break

label = network.gen_net.fc(label)
label = label.view(label.shape[0],1,initial_img_size,initial_img_size)
noise = torch.cat((noise,label),dim = 1)
output = network.gen_net.layer(noise)

with torch.no_grad():
    data_shape = output.shape
    for i in range(data_shape[0]):
        i1=transPIL(data[i])
        im1 = i1.save('E:\\cGAN\\fake\\f_'+str(i)+'.jpg','JPEG') #Location where the generated images should be stored
        i2=transPIL(output[i])
        im2 = i2.save('E:\\cGAN\\real\\r_'+str(i)+'.jpg','JPEG') #Location where true images(images of dataset) should be stored

'''
You need to create two folders and you should store images, generated images in one folder, original dataset images in another folder.
(Storing is done by the above code itself. You just need to add your desired location to store them. In above case, it's 
E:\\cGAN\\fake)
After storing, follow the procedure mentioned in readme
'''