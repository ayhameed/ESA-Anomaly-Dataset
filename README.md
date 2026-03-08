# ESA Mission 1 Anomaly Detection

PyTorch-based anomaly detection on ESA telemetry using an autoencoder and reconstruction error thresholding.

## Dataset

- Source: https://zenodo.org/records/15237121
- Scope used in this repo: **Mission 1 only** (`ESA-Mission1/`)
- Processed file used by the model: `esa_anomaly.csv`

## Project Structure

- `ESA_anomaly_detection_eda.ipynb` — data loading, preprocessing, and label generation
- `ESA_anomaly_detection_Model.ipynb` — training, thresholding, and evaluation plots
- `model/autoencoder.py` — autoencoder architecture
- `ESA-Mission1/` — Mission 1 telemetry and metadata

## Method (Short)

1. Merge selected telemetry channels.
2. Downsample and clean missing values (`ffill`/`bfill`).
3. Build binary anomaly labels from interval annotations.
4. Create temporal sliding windows.
5. Train autoencoder on normal data.
6. Compute reconstruction error and classify anomalies using a threshold.

## Metrics Reported

- Precision
- Recall
- F1-score
- Classification report
- Confusion matrix
- ROC curve (AUC)

## How to Run

1. Create/activate a Python environment.
2. Install dependencies:
   - `pandas`
   - `numpy`
   - `matplotlib`
   - `scikit-learn`
   - `torch`
   - `ipykernel` (for notebook kernel)
3. Run `ESA_anomaly_detection_eda.ipynb` to generate/update `esa_anomaly.csv`.
4. Run `ESA_anomaly_detection_Model.ipynb` to train and evaluate.

## Notes

- This repository intentionally focuses on **Mission 1** for reproducibility and manageable compute.
- Threshold can be fixed or selected automatically from the precision-recall curve.
