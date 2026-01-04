"""
Simple Streamlit dashboard scaffold for Brain MRI project.
Pages: Home, EDA, Samples, Inference, About

Run: streamlit run app.py
"""
import os
import glob
import random
from pathlib import Path
from typing import List

import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np

# Optional: lazy import tensorflow when needed
try:
    from tensorflow.keras.models import load_model
except Exception:
    load_model = None

# Paths
ROOT = Path(__file__).resolve().parent
EDA_PLOTS = ROOT / "eda_plots"
DATA_DIR = ROOT / "brain_tumor_organized"
MODEL_PATH = ROOT / "best_brain_tumor_model.keras"
LOG_PATH = ROOT / "predictions_log.csv"

st.set_page_config(page_title="Brain MRI EDA & Inference", layout='wide')

# Utilities
@st.cache_data
def read_counts_csv(csv_path: str = None):
    if csv_path is None:
        csv_path = EDA_PLOTS / "class_counts.csv"
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception:
        return None

@st.cache_data
def load_training_history(path: str = None):
    import json
    p = ROOT / (path or 'training_history.json')
    if p.exists():
        try:
            with open(p, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None

@st.cache_data
def load_model_metrics(path: str = None):
    import json
    p = ROOT / (path or 'model_metrics.json')
    if p.exists():
        try:
            with open(p, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    return None

@st.cache_resource
def load_keras_model(path: str = None):
    if path is None:
        path = MODEL_PATH
    if load_model is None:
        return None, "tensorflow not available"
    if not Path(path).exists():
        return None, f"Model not found at {path}"
    try:
        model = load_model(path)
        return model, None
    except Exception as e:
        return None, str(e)

def preprocess_image_for_model(img: Image.Image, target_size=(224, 224)) -> np.ndarray:
    # Ensure 3-channel
    img = img.convert('RGB')
    img = img.resize(target_size)
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, 0)
    return arr

def log_prediction(filename: str, predicted: str, prob: float):
    df = pd.DataFrame([[filename, predicted, float(prob)]], columns=["filename", "predicted", "prob"])
    if Path(LOG_PATH).exists():
        df.to_csv(LOG_PATH, mode='a', header=False, index=False)
    else:
        df.to_csv(LOG_PATH, index=False)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ['Home', 'EDA', 'Samples', 'Inference', 'About'])

# Home
if page == 'Home':
    st.title("Brain MRI — EDA & Model Dashboard")
    st.markdown("**Quick overview and links**")

    df = read_counts_csv()
    # Use two columns: left for metrics and counts, right for a compact plot
    left, right = st.columns([1, 1])

    if df is not None:
        total_images = int(df['total'].sum())
        with left:
            st.metric("Total images", total_images)
            with st.expander("Per-class counts (click to expand)"):
                st.dataframe(df)
            st.markdown("\n[Go to EDA page](#) to explore more interactive plots and histograms.")

            # Model metrics & training history
            metrics = load_model_metrics()
            history = load_training_history()
            if metrics:
                st.subheader('Model metrics ✅')
                try:
                    mdf = pd.DataFrame(list(metrics.items()), columns=['metric', 'value'])
                    st.table(mdf)
                except Exception:
                    st.json(metrics)
            else:
                st.info('No `model_metrics.json` found; run training notebook to generate model metrics.')

            if history:
                st.subheader('Training history (accuracy & loss) 📈')
                try:
                    hist_df = pd.DataFrame(history)
                    # Use Plotly if available for nicer rendering
                    try:
                        import plotly.express as px
                        fig = px.line(hist_df, x=hist_df.index, y=hist_df.columns, title='Training history')
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.line_chart(hist_df)
                except Exception:
                    st.write('Could not render training history')

        with right:
            dist_img = EDA_PLOTS / 'class_distribution_overall.png'
            # Show a smaller, well-scaled image to avoid overwhelming the layout
            if dist_img.exists():
                st.image(str(dist_img), caption='Overall class distribution', use_column_width=False, width=600)
            else:
                st.info('No distribution image found; run `eda_analysis.py` to generate plots.')

    else:
        st.info("No `class_counts.csv` found in `eda_plots/`. Run `eda_analysis.py` to generate it.")

# EDA page
elif page == 'EDA':
    st.header('Exploratory Data Analysis')
    df = read_counts_csv()
    if df is None:
        st.warning('Run EDA script first: `python eda_analysis.py`')
    else:
        st.subheader('Class counts')
        st.dataframe(df)
        # Interactive bar chart (Plotly)
        try:
            import plotly.express as px
            fig = px.bar(df, x='class', y='total', color='class', title='Overall class counts')
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.bar_chart(df.set_index('class')['total'])

        st.subheader('Saved plots')
        for fname in ['class_counts_per_split.png', 'samples_grid.png', 'image_size_distribution.png', 'pixel_intensity_hist.png']:
            p = EDA_PLOTS / fname
            if p.exists():
                st.image(str(p), caption=fname)

# Samples page
elif page == 'Samples':
    st.header('Browse Samples')
    df = read_counts_csv()
    classes = list(df['class']) if df is not None else []

    if classes:
        if 'show_use_column_width_notice' not in st.session_state:
            st.session_state['show_use_column_width_notice'] = True
        if st.session_state['show_use_column_width_notice']:
            left_col, right_col = st.columns([9,1])
            with left_col:
                st.warning("The `use_column_width` parameter has been deprecated and will be removed in a future release. Please utilize the `width` parameter instead.")
            with right_col:
                if st.button("Dismiss", key="dismiss_use_column_width_notice"):
                    st.session_state['show_use_column_width_notice'] = False
        cls = st.selectbox('Choose class', classes)
        n = st.slider('Number of samples', 1, 12, 5)
        # Prefer train split
        train_dir = DATA_DIR / 'train' 
        cls_dir = None
        if train_dir.exists() and (train_dir / cls).exists():
            cls_dir = train_dir / cls
        else:
            # fallback: check DATA_DIR/class
            if (DATA_DIR / cls).exists():
                cls_dir = DATA_DIR / cls
        if cls_dir is None or not cls_dir.exists():
            st.warning(f'No images found for class {cls}')
        else:
            imgs = list(cls_dir.glob('*.*'))
            imgs = [str(x) for x in imgs if x.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp')]
            if not imgs:
                st.warning('No image files found in folder')
            else:
                sampled = random.sample(imgs, min(n, len(imgs)))
                cols = st.columns(min(n, 6))
                for i, img_path in enumerate(sampled):
                    with cols[i % len(cols)]:
                        st.image(img_path, width=250)
                        st.caption(Path(img_path).name)
    else:
        st.info('No class counts found. Run EDA first.')

# Inference page
elif page == 'Inference':
    st.header('Run model inference')
    st.write('Upload an image or choose a sample from the dataset')

    uploaded = st.file_uploader('Upload image', type=['png', 'jpg', 'jpeg'])
    use_sample = st.checkbox('Use random sample from dataset (train/notumor balance)')

    model_loaded = False
    model = None
    model_status = None

    if st.button('Load model'):
        model, err = load_keras_model()
        if model is None:
            st.error(f'Could not load model: {err}')
        else:
            st.success('Model loaded')
            model_loaded = True

    # choose image
    image_to_run = None
    if uploaded is not None:
        image_to_run = Image.open(uploaded)
    elif use_sample:
        df = read_counts_csv()
        if df is None:
            st.warning('Run EDA to get class list')
        else:
            cls = st.selectbox('Sample class for inference', list(df['class']))
            train_dir = DATA_DIR / 'train'
            cls_dir = (train_dir / cls) if (train_dir / cls).exists() else (DATA_DIR / cls)
            imgs = list(cls_dir.glob('*.*')) if cls_dir.exists() else []
            imgs = [p for p in imgs if p.suffix.lower() in ('.jpg', '.png', '.jpeg', '.bmp')]
            if imgs:
                chosen = random.choice(imgs)
                image_to_run = Image.open(chosen)
                st.image(str(chosen), caption=chosen.name)

    if image_to_run is not None:
        st.image(image_to_run, caption='Input image', width=300)
        if model is None:
            st.info('Load a model first using the "Load model" button')
        else:
            # determine model input size if possible
            try:
                input_shape = model.input_shape
                h, w = input_shape[1], input_shape[2]
                target = (w or 224, h or 224)
            except Exception:
                target = (224, 224)
            x = preprocess_image_for_model(image_to_run, target)
            preds = model.predict(x)
            # If preds is single array
            if preds.ndim == 1 or (preds.ndim == 2 and preds.shape[0] == 1):
                probs = preds.ravel()
            else:
                probs = preds[0]

            df = read_counts_csv()
            classes = list(df['class']) if df is not None else [f'class_{i}' for i in range(len(probs))]

            # Map classes length
            if len(probs) != len(classes):
                # fallback: show raw probs
                st.write('Prediction output shape does not match class list; showing raw output')
                st.write(probs)
            else:
                out = sorted(zip(classes, probs), key=lambda x: -x[1])
                st.subheader('Top predictions')
                for cls, p in out[:5]:
                    st.write(f"{cls}: {p:.4f}")
                # log top1
                top1 = out[0]
                log_prediction(getattr(uploaded, 'name', str(random.random())), top1[0], float(top1[1]))

# About
elif page == 'About':
    st.header('About this dashboard')
    st.write('This is a lightweight Streamlit dashboard for the Brain MRI classification project.')
    st.write('- EDA charts and sample images are read from `eda_plots/`')
    st.write('- To run inference, place `best_brain_tumor_model.keras` at project root')
    st.write('\nRun locally:')
    st.code('streamlit run app.py')
