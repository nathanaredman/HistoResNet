import os
import torch
import matplotlib.pyplot as plt
import torchvision

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

plt.style.use('ggplot')

#image = Image.open('example.jpg')


preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(), 
])


#output_tensor = preprocess(image)

class CustomResNet(nn.Module):
    def __init__(self):
        super(CustomResNet, self).__init__()

    def forward(self, x):

        return x


class CustomDataSet(Dataset):
    def __init__(self):

    def __len__(self):

    def __getitem__(self, idx):


modeul = CustomResNet()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.001)