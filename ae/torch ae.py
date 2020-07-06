
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import cv2
import glob

batch_size = 256
learning_rate = 0.0002
num_epoch = 5



mnist_train = dset.MNIST("./", train=True, transform=transforms.ToTensor(), target_transform=None, download=True)
mnist_test = dset.MNIST("./", train=False, transform=transforms.ToTensor(), target_transform=None, download=True)


train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=batch_size, shuffle=True,num_workers=0,drop_last=True)
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=batch_size, shuffle=False,num_workers=0,drop_last=True)


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        self.encoder = nn.Linear(28*28,20)
        self.decoder = nn.Linear(20,28*28)   
                
    def forward(self,x):
        x = x.view(batch_size,-1)
        encoded = self.encoder(x)
        out = self.decoder(encoded).view(batch_size,1,28,28)
        return out


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = Autoencoder().to(device)
loss_func = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)




loss_arr =[]
for i in range(num_epoch):
    for j,[image,label] in enumerate(train_loader):
        x = image.to(device)
        
        optimizer.zero_grad()
        output = model.forward(x)
        loss = loss_func(output,x)
        loss.backward()
        optimizer.step()
        
    if j % 1000 == 0:
        print(loss)
        loss_arr.append(loss.cpu().data.numpy()[0])
        
out_img = torch.squeeze(output.cpu().data)
print(out_img.size())

for i in range(1):
    plt.subplot(1,2,1)
    plt.imshow(torch.squeeze(image[i]).numpy(),cmap='gray')
    plt.subplot(1,2,2)
    plt.imshow(out_img[i].numpy(),cmap='gray')
    plt.show()       
        
    
# def make_batch(img_list, batch_size):
#     i = 0
#     train_dataset = []
#     tmp = []
#     for each in img_list:
#         if i == batch_size:
#             i = 0
#             train_dataset.append(tmp)
#             tmp = []
#         img = cv2.imread(each, cv2.IMREAD_GRAYSCALE)
#         img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        
#         img = np.moveaxis(img, -1, 0)
#         tmp.append(img.astype("float32")/255.0)
#         i += 1
#     # import ipdb; ipdb.set_trace()
#     # Y_data = np.vstack(train_dataset)
#     # Z_data = copy.deepcopy(Y_data)
#     # Z_data = (Z_data - Z_data.mean()) / Z_data.std()
#     return train_dataset


# train_imgs = glob.glob("C:/Users/CT_NT_CNY/Desktop/python test1/mel ae/image/image cut/*.png")
# test_imgs = glob.glob("C:/Users/CT_NT_CNY/Desktop/python test1/mel ae/image/image cut/*.png")
# train_loader = make_batch(train_imgs, 128)
# test_loader = make_batch(test_imgs, 128)


# model.forward(x)




# with torch.no_grad():    
#   for i in range(1):
#       for j,[image,label] in enumerate(test_loader):
#           x = image.to(device)

#           optimizer.zero_grad()
#           output = model.forward(x)

#       if j % 1000 == 0:
#           print(loss)

# out_img = torch.squeeze(output.cpu().data)
# print(out_img.size())

# for i in range(2):
#     plt.subplot(1,2,1)
#     plt.imshow(torch.squeeze(image[i]).numpy(),cmap='gray')
#     plt.subplot(1,2,2)
#     plt.imshow(out_img[i].numpy(),cmap='gray')
#     plt.show()    
        
        
        
