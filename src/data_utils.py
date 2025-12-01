import os
import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

# Image transformations (resize, crop, normalize)
image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    # Normalize with ImageNet mean/std
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class WhiteBalanceImageDataset(Dataset):
    """Dataset for loading images and optional labels from a CSV."""
    def __init__(self, img_dir, csv_file, transform=None, train=True):
        """
        img_dir: path to folder with images
        csv_file: path to CSV with 'id_global' and features; if train=True, also 'Temperature','Tint'
        """
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform
        self.train = train
        # If training, ensure label columns exist
        if self.train and not {'Temperature','Tint'}.issubset(self.df.columns):
            raise ValueError("Training CSV must contain Temperature and Tint columns.")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Load image
        row = self.df.iloc[idx]
        img_name = row['id_global']  # assume image filenames match id_global
        img_path = os.path.join(self.img_dir, f"{img_name}.tiff")
        image = Image.open(img_path).convert('RGB')  # ensure 3-channel
        if self.transform:
            image = self.transform(image)
        # Load metadata features as needed (we drop or handle them elsewhere)
        # ...
        # Load labels if training
        if self.train:
            temp = row['Temperature']
            tint = row['Tint']
            target = [temp, tint]
            return image, torch.tensor(target, dtype=torch.float32)
        else:
            return image
