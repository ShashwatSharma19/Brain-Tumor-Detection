# Dashboard spec — Brain MRI

## Pages

1. Home
   - Project summary
   - Total images metric
   - Model metrics (accuracy & loss) and training history plot
   - Quick links to run EDA / Re-generate plots

2. EDA
   - Table of per-class counts (CSV)
   - Interactive bar/stacked charts per split
   - Histograms for width/height and pixel intensity
   - PNG previews (samples, distribution)

3. Samples
   - Pick class and n samples
   - Grid view and full-size viewer
   - Metadata: filename, shape

4. Inference
   - Upload image or pick sample
   - Load model button
   - Show predictions and top-k probabilities
   - Save prediction log
   - (Optional) Grad-CAM

5. About
   - Repo links, run & deploy instructions

## Deploy
- Streamlit Community Cloud (free) — simple: `requirements.txt` and `app.py` in repo root
- Docker deploy to cloud (Heroku/Render/Azure/AWS ECS)

## Notes
- Keep inference synchronous and light-weight for demo
- Add caching for model to avoid reload
