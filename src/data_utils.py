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

# src/data_utils.py (continued)

from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_metadata(df, is_train=True, scaler=None, label_encoders=None):
    """
    Cleans and encodes metadata DataFrame.
    If is_train=True, fits encoders/scaler and returns them.
    If is_train=False, uses provided scaler/encoders.
    """
    # Copy to avoid modifying original
    df = df.copy()
    # Example: drop columns not needed (e.g., id_global after extraction)
    df = df.drop(columns=['id_global'], errors='ignore')
    # Separate target if present
    if is_train:
        y = df[['Temperature','Tint']].values
        df = df.drop(columns=['Temperature','Tint'], errors='ignore')
    else:
        y = None
    
    # Identify numeric vs categorical columns
    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    cat_cols = df.select_dtypes(include=['object','category']).columns.tolist()
    
    # Fill missing values
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in cat_cols:
        df[col].fillna('Unknown', inplace=True)
    
    # Encode categorical: use label encoding for simplicity
    if label_encoders is None:
        label_encoders = {}
    for col in cat_cols:
        if is_train:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            # Use existing encoder, handling unseen values if any
            le = label_encoders.get(col)
            if le:
                df[col] = le.transform(df[col])
            else:
                df[col] = 0
    
    # Scale numeric features
    if scaler is None:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    else:
        df[num_cols] = scaler.transform(df[num_cols])
    
    X = df.values  # final feature matrix
    return X, y, scaler, label_encoders
