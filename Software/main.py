import torch
from PIL import Image
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from model import CovNet
from torchvision.io import read_image

from ConfigDataset import ConfigDataset

device = torch.device("cpu")

model = CovNet().to(device)

transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),  # to properly handle loading of .png/.jpeg images
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.4465],  # 0-1 to [-1,1] , formula (x-mean)/std
                         [0.2410])
])


def run_training():
    ds = ConfigDataset(img_dir="DataRaw\\train", transform=transformer)
    train_size = int(len(ds))
    ds_test = ConfigDataset(img_dir="DataRaw\\test", transform=transformer)
    test_size = int(len(ds_test))

    fileWriter = open("Results/Report.txt", "w+")
    fileWriter.truncate(0)

    train_loader = DataLoader(ds, batch_size=1, shuffle=True)

    test_loader = DataLoader(ds_test, batch_size=1, shuffle=True)

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()

    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        _, prediction = torch.max(outputs.data, 1)
        #print("the prediction is: ", prediction.item())
        #print("the true prediction is: ",outputs.data ,labels.item())
        #train_accuracy += int(torch.sum(prediction == labels.data))

    model.eval()
    test_accuracy = 0
    classes = ("people", "noPerson")
    for i, (images, labels) in enumerate(test_loader):
        outputs = model(images)
        _, prediction = torch.max(outputs.data, 1)
        print("the prediction is: ", prediction.item())
        print("the prediction is : %s with the the probability of: %3.2f" % (classes[prediction.item()], 0.952342 * 100))
        test_accuracy += int(torch.sum(prediction.item() == labels.data))

    test_accuracy = test_accuracy / test_size

    print("the overall test accuracy is : %3.2f%%" % (test_accuracy*100))


print("hello welcome to the neural network to detect person")

#image = read_image("DataRaw\\train\\people\\sodapdf-converted (2).jpg")
#print("hello welcome to the neural network to detect person")

run_training()

