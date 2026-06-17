import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch, requests, io

def message():
    st.error("Please reload the application and allow 3–4 minutes (6-7 minutes if your internet speed is slow i.e below 40 MBPS) for the dashboard to fully initialize the Baseline Market. Avoid changing sidebar filters during this process.")
    st.warning("The filter feature compares your 'Custom/Filtered Market' selection against the 'Baseline Market'. For accurate results, let the dashboard complete its loading cycle. You will receive toast notifications as the dashboard becomes ready; until then, please refrain from adjusting filters.")

def assign_rank_group(position):
    if 1 <= position <= 10:
        return 'Top 10'
    elif 11 <= position <= 50:
        return 'Top 11-50'
    else:
        return 'Other'

@st.cache_data
def validate_and_preprocess(df):
    # Standardize artist names
    df['artist'] = df['artist'].str.lower().str.strip()

    # Split multi-artist collaborations
    df['artist'] = df['artist'].astype(str).apply(lambda x: [a.strip() for a in x.split('&')])
    df = df.explode('artist')

    # Track collaborations
    track_collaborations = df.groupby(['date', 'song', 'position']).agg(
        num_artists=('artist', 'nunique')
    ).reset_index()
    track_collaborations['is_collaboration'] = track_collaborations['num_artists'] > 1

    df_merged = pd.merge(
        df,
        track_collaborations[['date', 'song', 'position', 'is_collaboration']],
        on=['date', 'song', 'position'],
        how='left'
    )

    # Rank group
    df_merged['rank_group'] = df_merged['position'].apply(assign_rank_group)

    # Convert date
    df_merged['date'] = pd.to_datetime(df_merged['date'], dayfirst=True)

    return df_merged

# Helper function for metrics
def get_metrics(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    metrics_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
    metrics_df = metrics_df.drop(labels=['accuracy','macro avg','weighted avg'])
    metrics_df.rename(index={'0':'Class 0 (Not Top 10)','1':'Class 1 (Top 10)'}, inplace=True)
    return acc, metrics_df

def build_comparison_df(metrics_no_eng_df, metrics_eng_df, model_name):
        comparison_data = []
        metrics_no_eng_dict = metrics_no_eng_df.to_dict('index')
        metrics_eng_dict = metrics_eng_df.to_dict('index')
    
        for class_name_key in metrics_no_eng_dict:
            metrics = metrics_no_eng_dict[class_name_key]
            comparison_data.append({'Model': f'{model_name} (No Features)', 'Class': class_name_key, 'Metric': 'Precision', 'Score': metrics['precision']})
            comparison_data.append({'Model': f'{model_name} (No Features)', 'Class': class_name_key, 'Metric': 'Recall', 'Score': metrics['recall']})
            comparison_data.append({'Model': f'{model_name} (No Features)', 'Class': class_name_key, 'Metric': 'F1-score', 'Score': metrics['f1-score']})
    
        for class_name_key in metrics_eng_dict:
            metrics = metrics_eng_dict[class_name_key]
            comparison_data.append({'Model': f'{model_name} (Engineered Features)', 'Class': class_name_key, 'Metric': 'Precision', 'Score': metrics['precision']})
            comparison_data.append({'Model': f'{model_name} (Engineered Features)', 'Class': class_name_key, 'Metric': 'Recall', 'Score': metrics['recall']})
            comparison_data.append({'Model': f'{model_name} (Engineered Features)', 'Class': class_name_key, 'Metric': 'F1-score', 'Score': metrics['f1-score']})
    
        return pd.DataFrame(comparison_data)

# Conceptual definition of major genres
major_genres = ['Pop', 'Rock', 'Hip-Hop/Rap', 'Jazz', 'Country',
                'Classical', 'Dance', 'R&B/soul', 'Electronic/EDM', 'Folk',
                'Metal', 'Blues', 'Reggae', 'Instrumental', 'Indie',
                'Gospel', 'Punk', 'Latin', 'Afrobeats', 'World Music']

# Load CLIP model from openai
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def predict_genre_from_image_ai_conceptual(image_url):
    """
    Predict genre from album cover URL using CLIP zero-shot classification.
    Always returns one of the major_genres.
    """
    if pd.isna(image_url) or not isinstance(image_url, str) or not image_url.startswith('http'):
        return 'Unknown'
    try:
        response = requests.get(image_url, timeout=5)
        response.raise_for_status()
        image = Image.open(io.BytesIO(response.content)).convert("RGB")

        # Compare image against all genre prompts
        # inputs = processor(text=major_genres, images=image, return_tensors="pt", padding=True)
        text_inputs = processor.tokenizer(
            major_genres,
            padding=True,
            return_tensors="pt"
        )
        image_inputs = processor.image_processor(
            images=image,
            return_tensors="pt"
        )

        inputs = {
            **text_inputs,
            **image_inputs
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)

        predicted_idx = probs.argmax(-1).item()
        return major_genres[predicted_idx]
    except Exception:
        return "Unknown"