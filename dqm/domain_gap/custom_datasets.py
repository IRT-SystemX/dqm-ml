from torch.utils.data import Dataset
from PIL import Image
import os


# Définir une classe de Dataset personnalisée pour charger les images à partir du fichier .txt
class ImagesWithLabelsDataset(Dataset):
    def __init__(self, txt_file, transform=None):
        self.txt_file = txt_file
        self.transform = transform

        # Lire le fichier .txt et stocker les chemins d'images et les labels dans des listes
        self.images = []
        self.labels = []
        with open(txt_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                image_path, label = line.strip().split(" ")
                self.images.append(image_path)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert("RGB")  # Ouvrir l'image en mode RGB
        if self.transform:
            image = self.transform(image)
        return image, label


class ImagesFromFolderDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.images = os.listdir(self.folder_path)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(os.path.join(self.folder_path, self.images[idx]))
        if self.transform:
            image = self.transform(image)
        return image
