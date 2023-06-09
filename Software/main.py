import torch
from PIL import Image
import numpy as np
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from model import CovNet

from ConfigDataset import ConfigDataset

device = torch.device("cpu")

model = CovNet.to(device)


transformer = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),  # to properly handle loading of .png/.jpeg images
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.4907, 0.4465, 0.4261],  # 0-1 to [-1,1] , formula (x-mean)/std
                         [0.2397, 0.2410, 0.2505])
])



def run_training():
    ds = ConfigDataset(img_dir="DataRaw/test", transform=transformer)
    train_size = int(len(ds))

    fileWriter = open("Results\Report.txt", "w+")
    fileWriter.truncate(0)

    train_loader = DataLoader(ds,batch_size=10,shuffle=True)

    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
    loss_function = nn.CrossEntropyLoss()

    # Evaluation and training on training dataset
    model.train()
    train_accuracy = 0.0
    train_loss = 0.0













print("hello world this is wild")
