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
    ds = ConfigDataset(img_dir="DataRaw\\test", transform=transformer)
    train_size = int(len(ds))
    print(train_size)
    ds_test = ConfigDataset(img_dir="DataRaw\\train", transform=transformer)
    test_size = int(len(ds_test))

    fileWriter = open("Results/Report.txt", "w+")
    fileWriter.truncate(0)

    train_loader = DataLoader(ds, batch_size=1, shuffle=True)

    test_loader = DataLoader(ds_test, batch_size=1, shuffle=True)

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = nn.BCELoss()

    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_index =0
    train_loss = 0.0
    num_epoch =10
    for epoch in range(num_epoch):
        train_index =0
        train_accuracy = 0.0
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            floatLabel=labels.float()
            loss = loss_function(outputs[0],  floatLabel)
            loss.backward()
            optimizer.step()
            prediction = 0
            if(outputs.item()>0.5):
                prediction =1

            #_, prediction = torch.max(outputs.data, 1)
            #print(prediction, labels.data)

            #print("the prediction is: ", prediction.item())
            #print("the true prediction is: ",outputs.data ,labels.item())
            if( prediction == labels.item()):
                train_accuracy += 1
            train_index+=1
        print(train_index)
        print(f"epoch {epoch} the prediction is: %f" % (train_accuracy/train_index))

    torch.save(model.state_dict(), 'model_1.pth')


    model.eval()
    test_accuracy = 0
    classes = ("people", "noPerson")
    for i, (images, labels) in enumerate(test_loader):
        outputs = model(images)
        prediction = 0
        if (outputs.item() > 0.5):
            prediction = 1
        print("the expected is: ", labels.item())
        print("the prediction is : %s with the the probability of: %3.2f %%" % (classes[prediction], 100-(outputs.item()*100)))
        if (prediction == labels.item()):
            test_accuracy += 1

    test_accuracy = test_accuracy / test_size

    print("the overall test accuracy is : %3.2f%%" % (test_accuracy*100))

    loop = True
    while (loop):
        print("please enter a word or 1 to exit")
        aWord = input()
        if (aWord == "1"):
            print("goodbye")
            break
        else:

            print("enter the path to the desired image\n>")
            path = input()
            print(path)
            labels =input("enter the associated label either people or noPerson")


            ds_singleIntance = ConfigDataset(transform=transformer,singleImageInstance=(path,labels))
            single_loader = DataLoader(ds_singleIntance, batch_size=1, shuffle=True)

            for i, (images, labels) in enumerate(single_loader):

                outputs = model(images)

                _, prediction = torch.max(outputs.data, 1)
                print("the prediction is : %s with the the probability of: %3.2f" % (
                classes[prediction.item()], 0.952342 * 100))


print("hello welcome to the neural network to detect person")

#image = read_image("DataRaw\\train\\people\\sodapdf-converted (2).jpg")
#print("hello welcome to the neural network to detect person")

run_training()

