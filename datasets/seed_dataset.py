import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SeedDataset(Dataset):
    def __init__(self, data_path: str, setname: str, augment: bool):
        self.data_path = os.path.join(data_path, setname)
        self.augment = augment

        self.data = []
        self.labels = {'good': 0, 'bad': 1}  # Assuming only two labels as per your description

        for label in os.listdir(self.data_path):
            for img_file in os.listdir(os.path.join(self.data_path, label)):
                self.data.append((os.path.join(self.data_path, label, img_file), self.labels[label]))

        self.transform = self.get_transform()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_transform(self):
        transforms_list = [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ]
        if self.augment:
            transforms_list = [
                transforms.RandomResizedCrop(32),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
            ] + transforms_list

        return transforms.Compose(transforms_list)
