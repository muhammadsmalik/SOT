# datasets/seed_fs.py
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class SeedFS(Dataset):
    def __init__(self, data_path: str, setname: str, backbone: str, augment: bool):
        self.data_path = os.path.join(data_path, 'seed_fs', setname)
        self.augment = augment
        self.image_size = 256
        self.data = []
        self.label = []
        self.labels_map = {'good': 0, 'bad': 1}

        # Load data
        self._load_data()

        # Define transforms
        self.transform = self._get_transforms()

    def _load_data(self):
        for label_dir in os.listdir(self.data_path):
            # Convert directory name to lowercase to ensure case-insensitivity
            label = label_dir.lower()  
            label_path = os.path.join(self.data_path, label_dir)
            if label in self.labels_map:  # Check if the label is in labels_map
                for image_name in os.listdir(label_path):
                    self.data.append(os.path.join(label_path, image_name))
                    self.label.append(self.labels_map[label])
            else:
                print(f"Warning: Label '{label_dir}' not recognized and will be skipped.")


    def _get_transforms(self):
        mean = [0.485, 0.456, 0.406]  # ImageNet means
        std = [0.229, 0.224, 0.225]  # ImageNet stds
        normalize = transforms.Normalize(mean=mean, std=std)
        
        if self.augment:
            return transforms.Compose([
                transforms.RandomResizedCrop(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        else:
            return transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                normalize,
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path, label = self.data[idx], self.label[idx]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
