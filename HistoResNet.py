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