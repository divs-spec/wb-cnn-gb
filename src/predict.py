
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from data_utils import WhiteBalanceImageDataset, image_transforms, preprocess_metadata
from model import ResNet18Regressor
import joblib

# Paths
VAL_IMG_DIR = 'data/Validation'
VAL_CSV = 'data/Validation/sliders_inputs.csv'
CNN_MODEL_PATH = 'cnn_model.pth'
META_MODEL_PATH = 'meta_model.pkl'
OUTPUT_CSV = 'submission.csv'

# Load validation IDs
df_val = pd.read_csv(VAL_CSV)
val_ids = df_val['id_global'].values

# ---- Image predictions ----
val_image_dataset = WhiteBalanceImageDataset(VAL_IMG_DIR, VAL_CSV, transform=image_transforms, train=False)
val_image_loader = DataLoader(val_image_dataset, batch_size=16, shuffle=False, num_workers=4)

# Load CNN model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model = ResNet18Regressor(pretrained=False).to(device)
cnn_model.load_state_dict(torch.load(CNN_MODEL_PATH, map_location=device))
cnn_model.eval()

all_image_preds = []
with torch.no_grad():
    for images in val_image_loader:
        images = images.to(device)
        preds = cnn_model(images)  # shape [batch,2]
        all_image_preds.append(preds.cpu().numpy())
all_image_preds = np.vstack(all_image_preds)  # shape (N_val, 2)

# ---- Metadata predictions ----
# Preprocess metadata (use saved scaler/encoders)
meta_data = pd.read_csv(VAL_CSV)
X_meta_val, _, _, _ = preprocess_metadata(meta_data, is_train=False, 
                                         scaler=joblib.load(META_MODEL_PATH)['scaler'],
                                         label_encoders=joblib.load(META_MODEL_PATH)['encoders'])
loaded = joblib.load(META_MODEL_PATH)
meta_model = loaded['model']
meta_preds = meta_model.predict(X_meta_val)  # shape (N_val, 2)

# ---- Combine predictions ----
# Here we simply average the CNN and metadata predictions
final_preds = (all_image_preds + meta_preds) / 2.0

# Round to nearest integer as required
final_preds = np.rint(final_preds).astype(int)  # shape (N_val, 2)

# ---- Generate submission CSV ----
output_df = pd.DataFrame({
    'id_global': val_ids,
    'Temperature': final_preds[:,0],
    'Tint': final_preds[:,1]
})
output_df.to_csv(OUTPUT_CSV, index=False)
print(f"Submission saved to {OUTPUT_CSV}")
