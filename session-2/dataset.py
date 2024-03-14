import os

import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class MyDataset(Dataset):

    def __init__(self, images_path, labels_path, transform=None):

        super(MyDataset, self).__init__()
        self.images_path = images_path
        self.df_labels = pd.read_csv(labels_path)
        self.transform = transform

    def __len__(self):
        return len(self.df_labels)

    def __getitem__(self, idx):

        label_info = self.df_labels[idx]
        image_name = f'input_{label_info[0]}_{label_info[1]}_{label_info[2]}.jpg'
        image_dir = os.path.join(self.images_path, image_name)


        trans = self.transform
        img = None

        if isinstance(trans, transforms.Compose):
            img = trans(Image.open(image_dir))
        else:
            trans = transforms.Compose(
                #resize
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
            img = trans(Image.open(image_dir))

        # return tensor imagen (Pil), label
        return img, label_info[2]
    


if __name__ == "__main__":

    print("hola")