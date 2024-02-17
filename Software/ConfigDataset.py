import glob
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image


class ConfigDataset(Dataset):
    classes = ("people", "noPerson")

    def __init__(self, img_dir="DataRaw/test", transform=None, target_transform=None, singleImageInstance=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.singleImageInstance = singleImageInstance
        self.img_labels = self.load_images(img_dir)

    def load_images(self, img_dir):
        records = []
        if self.singleImageInstance == None:
            for class_idx, label in enumerate(self.classes):
                for image_path in glob.iglob(f'{img_dir}\\{label}\\*.jpg'):
                    records.append((image_path, class_idx))

        else:
            records.append((self.singleImageInstance[0], self.classes.index(self.singleImageInstance[1])))

        return pd.DataFrame(records)

    def get_class_name(self, label):
        """
        Returns classname from label

        example:  get_class_name(0) -> "Cloth"
        """
        return self.classes[label]

    def get_label(self, class_name):
        """
        Returns label from class_name

        example: get_label("Surgical") -> 2
        """
        return self.classes.index(class_name)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = read_image(self.img_labels.iloc[idx, 0])
        label = self.img_labels.iloc[idx, 1]


        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label
