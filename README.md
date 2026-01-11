# brain-tumor-detection

**Tagline:** A MobileNetV2-based CNN pipeline for classifying brain MRI scans (glioma, meningioma, pituitary, no tumor).

---

## Table of Contents
- [Project Overview](#project-overview)
- [Results & Findings](#results--findings)
- [Data](#data)
- [Reproducing the Metrics](#reproducing-the-metrics)
- [How the Values Were Obtained](#how-the-values-were-obtained)
- [Usage](#usage)
- [Recommended Next Steps](#recommended-next-steps)
- [Files of Interest](#files-of-interest)
- [License & Contact](#license--contact)

---

## Project Overview
`brain-tumor-detection` is a compact, reproducible repository for detecting and classifying brain tumors from MRI images using transfer learning (MobileNetV2). The repository includes data organization, augmentation, training (frozen base + fine-tuning), checkpointing, evaluation and a minimal Streamlit dashboard.

---
## DashBoard 
![Dashboard Screenshot](https://github.com/ShashwatSharma19/Brain-Tumor-Detection/blob/c193cadfd74c71900bef3d0d927db829959aaade/brain-mri.png)
![Dashboard EDA SC](https://github.com/ShashwatSharma19/Brain-Tumor-Detection/blob/c193cadfd74c71900bef3d0d927db829959aaade/EDA-brainmri.png)
![Dashboard EDA 2 SC](https://github.com/ShashwatSharma19/Brain-Tumor-Detection/blob/c193cadfd74c71900bef3d0d927db829959aaade/EDA-brainmri2.png)
![Dashboard EDA 3 SC](https://github.com/ShashwatSharma19/Brain-Tumor-Detection/blob/c193cadfd74c71900bef3d0d927db829959aaade/EDA-brainmri3.png)
![Dashboard EDA 4 SC](https://github.com/ShashwatSharma19/Brain-Tumor-Detection/blob/c193cadfd74c71900bef3d0d927db829959aaade/EDA-brainmri4.png)



## Results & Findings ✅
- **Final Test Accuracy:** **96.50%** (evaluated on held-out `test/` split using the saved best model `best_brain_tumor_model.keras`).
- **Best Validation Accuracy during training:** **~85.27%** (Phase 1 best epoch), and improved during fine-tuning to values > 88% at times.
- Example epoch logs and checkpoint updates are stored in the executed `model.ipynb`.

---

## Data
- Data directory: `brain_tumor_organized/` with `train/`, `val/`, `test/` splits organized by class.
- Use `eda_plots/class_counts.csv` to inspect class distributions.
- ![Dashboard samples](https://github.com/ShashwatSharma19/Brain-Tumor-Detection/blob/c193cadfd74c71900bef3d0d927db829959aaade/brain-mri%20samples.png)
- ![Dashboard inference](https://github.com/ShashwatSharma19/Brain-Tumor-Detection/blob/c193cadfd74c71900bef3d0d927db829959aaade/brain-mri%20inference.png)
- ![DashBoard about](https://github.com/ShashwatSharma19/Brain-Tumor-Detection/blob/c193cadfd74c71900bef3d0d927db829959aaade/brain-mri%20about.png)

---

## Reproducing the Metrics (quick steps) 🔧
1. Create & activate a virtual environment, then install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   pip install -r requirements.txt
   ```
2. Open and run `model.ipynb` in Jupyter. Training runs in two phases:
   - Phase 1: Train classifier head with base MobileNetV2 frozen
   - Phase 2: Fine-tune the last ~30 layers of the base model
3. The best model is saved automatically as `best_brain_tumor_model.keras` (checkpointed on `val_accuracy`).
4. Final evaluation metrics are printed and saved to `model_metrics.json`; training history is saved to `training_history.json` and plot to `training_history.png`.

---

## How the Values Were Obtained 📊
- Training uses Keras `ImageDataGenerator` flows and `Model.fit(...)`; metrics recorded per epoch include accuracy, loss, precision, and recall (both train and validation).
- The final Test Accuracy (96.50%) is reported by running `model.evaluate(test_gen)` on the saved best checkpoint.

---

## Usage
- Quick Inspect (Streamlit dashboard):
  ```bash
  streamlit run app.py
  ```
- Inference example:
  ```python
  import tensorflow as tf
  model = tf.keras.models.load_model('best_brain_tumor_model.keras')
  preds = model.predict(my_preprocessed_images)
  ```

---

## Recommended Next Steps 💡
- Verify the test split and ensure no data leakage to confirm the 96.50% accuracy is robust.
- Inspect per-class metrics and confusion matrix for bias analysis.
- Try stronger augmentation, focal loss, or cross-validation if improvement is needed.

---

## Files of Interest
- `model.ipynb` — training & evaluation (source of the metrics)
- `best_brain_tumor_model.keras` — saved best model
- `training_history.png`, `training_history.json`, `model_metrics.json` — training artifacts
- `app.py` — dashboard that reads `training_history.json`

