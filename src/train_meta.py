
import pandas as pd
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
from data_utils import preprocess_metadata

# Paths
META_CSV = 'data/Train/sliders.csv'
MODEL_META_PATH = 'meta_model.pkl'

# Load metadata CSV
df = pd.read_csv(META_CSV)

# Preprocess metadata
X_meta_train, y_train, scaler, label_encoders = preprocess_metadata(df, is_train=True)

# Initialize LightGBM regressor (default params, can be tuned)
gbm = LGBMRegressor(n_estimators=100, random_state=42)
multi_gbm = MultiOutputRegressor(gbm)

# Train
multi_gbm.fit(X_meta_train, y_train)

# Save model (using joblib or pickle)
import joblib
joblib.dump({'model': multi_gbm, 'scaler': scaler, 'encoders': label_encoders}, MODEL_META_PATH)
