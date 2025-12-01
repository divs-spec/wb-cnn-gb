# Hybrid CNN + Gradient-Boosting Pipeline for White Balance Prediction

We build a multimodal pipeline that ingests both image data and tabular metadata to predict Temperature and Tint (white balance) values. The approach fuses deep image features from a CNN (e.g. ResNet) with a gradient-boosting model on metadata. This hybrid design has been shown to outperform single-modality models for this task.
We use Mean Absolute Error (MAE) as the metric (rounded predictions to integers as required).

Key steps:

Preprocess images and metadata separately.

**1. Image model:** Use a pretrained CNN (ResNet-18) as a feature extractor, fine-tuned for regression (2 outputs).

**2. Metadata model:** Use a tree boosting regressor (LightGBM/XGBoost) on tabular features.

**3. Fusion:** Combine outputs (or features) from both models for final prediction, e.g. by averaging or a small ensemble.

**4. Evaluation:** Optimize for MAE and round predictions to nearest integer.

**5. Explainability:** Leverage feature importances from the boosting model for insight.

# Project Structure

The code is organized into modular scripts:

**- data_utils.py** – utilities to load and preprocess images and metadata.

**- model.py** – defines the PyTorch CNN architecture (e.g. ResNet-based regression).

**- train_cnn.py** – trains the CNN on images + optionally metadata, saves model weights.

**- train_meta.py** – trains the LightGBM model on metadata, saves model.

**- predict.py** – loads trained models, makes predictions on validation set, combines them, and writes the submission CSV.

**- requirements.txt** – pip-installable dependencies.
