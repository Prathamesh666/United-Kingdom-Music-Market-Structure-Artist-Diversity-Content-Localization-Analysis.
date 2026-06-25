import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch, requests, io, urllib.parse, base64
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
    
# Spotify credentials (replace with your own)
CLIENT_ID = st.secrets.get("Client_ID")
CLIENT_SECRET = st.secrets.get("Client_Secret")
import time
@st.cache_data(ttl=3600)
def get_spotify_token_cached():
    return get_spotify_token()

# Get Spotify access token
def get_spotify_token():
    auth_url = "https://accounts.spotify.com/api/token"
    auth_response = requests.post(auth_url, {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    })
    return auth_response.json()["access_token"]

def search_spotify_track(song, artist, headers):
    query = urllib.parse.quote(f"{song} {artist}")
    url = f"https://api.spotify.com/v1/search?q={query}&type=track&limit=1"
    response = requests.get(url, headers=headers)
    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 5))
        st.warning(f"Rate limit hit. Retrying after {retry_after} seconds...")
        time.sleep(retry_after)
        response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Spotify API error:", response.status_code, response.text)
        return None
    try:
        results = response.json()
    except ValueError:
        print("Response was not JSON:", response.text[:200])  # show first 200 chars
        return None

    items = results.get("tracks", {}).get("items", [])
    if items:
        track_id = items[0]["id"]
        preview_url = items[0].get("preview_url")  # may be None
        return track_id, preview_url
    return None

def get_youtube_video_id(song: str, artist: str, api_key: str) -> str:
    """
    Search YouTube for the official music video of a song + artist.
    Returns the first matching video ID or None.
    """
    query = f"{song} {artist} official music video"
    url = "https://www.googleapis.com/youtube/v3/search"
    
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "videoCategoryId": "10",   # Music category
        "maxResults": 1,
        "key": api_key
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        items = data.get("items", [])
        if items:
            return items[0]["id"]["videoId"]
    return "None"

REDIRECT_URI = "https://um-unitedkingdommusicmarketanalysisdashboard.streamlit.app/"

# Exchange authorization code for access token
def get_token(code: str):
    url = "https://accounts.spotify.com/api/token"
    payload = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": REDIRECT_URI,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    response = requests.post(url, data=payload)
    st.info(f"Token response: {response.json()}")  # Debugging
    return response.json()

# Refresh access token using refresh_token
def refresh_access_token(refresh_token: str):
    url = "https://accounts.spotify.com/api/token"
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    response = requests.post(url, data=payload)
    data = response.json()
    st.info(f"Refresh response: {data}")  # Debugging
    return data

# Get current user info (id + product type)
def get_spotify_user_info(token: str):
    url = "https://api.spotify.com/v1/me"
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to fetch user info: {response.text}")
        return None

# Build playlist description dynamically
def build_playlist_description(start_date, end_date, selected_artists, collaboration_choice, selected_album_types,
                                duration_range, selected_popularity, selected_genres, is_any_filter_different):
    if not is_any_filter_different:
        return "Generated from Streamlit dashboard"

    description_parts = [ "Curated with your filters: Date Range, Artists, Track Type, Album Type, Duration, Popularity, Genres.",
        f"Date Range: {start_date} to {end_date}",
        f"Track Type: {collaboration_choice}",
        f"Album Types: {', '.join(selected_album_types) if selected_album_types else 'All'}",
        f"Duration: {duration_range[0]}–{duration_range[1]} minutes",
        f"Popularity: {selected_popularity[0]}–{selected_popularity[1]}"
    ]
    print(description_parts) # debugging

    return "Playlist generated with filters → " + " | ".join(description_parts)

# Create playlist (modern endpoint)
def create_spotify_playlist(token, name, description):
    url = "https://api.spotify.com/v1/me/playlists"
    payload = {"name": name, "description": description, "public": False}
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(url, json=payload, headers=headers)
    st.info(f"Playlist info: {response.json()}")
    return response.json()

def upload_playlist_cover(playlist_id, image_path, token):
    # Convert PNG → JPEG in memory
    img = Image.open(image_path).convert("RGB")
    img = img.resize((640, 640))
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    encoded_image = base64.b64encode(buffer.getvalue())

    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/images"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "image/jpeg"
    }
    response = requests.put(url, data=encoded_image, headers=headers)

    if response.status_code == 202:
        st.success("🎨 Playlist cover uploaded successfully!")
    else:
        st.error(f"Failed to upload cover: {response.json()}")

def add_tracks_to_playlist(playlist_id, track_uris, token):
    url = f"https://api.spotify.com/v1/playlists/{playlist_id}/items" # Since /tracks is deprecated
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }

    total = len(track_uris)
    num_batches = (total // 100) + (1 if total % 100 else 0)

    progress_bar = st.progress(0)
    progress_text = st.empty()

    for batch_idx, i in enumerate(range(0, total, 100), start=1):
        chunk = track_uris[i:i+100]
        payload = {"uris": chunk}
        import json
        st.write("Payload being sent:", json.dumps({"uris": track_uris[:5]}, indent=2))
        st.write("Sending chunk:", payload)

        response = requests.post(url, json=payload, headers=headers)

        if response.status_code not in (200, 201):
            st.error(f"Failed to add tracks: {response.json()}")
            return response.json()

        percent_complete = int((batch_idx / num_batches) * 100)
        progress_bar.progress(percent_complete)
        progress_text.text(f"Uploading batch {batch_idx}/{num_batches} ({len(chunk)} tracks)...")

    progress_bar.progress(100)
    progress_text.text("✅ All tracks uploaded successfully!")
    return {"status": "success"}

def create_playlist_from_dataframe(unique_songs, start_date, end_date, selected_artists, collaboration_choice, selected_album_types, duration_range,
                                selected_popularity, selected_genres, is_any_filter_different):
    query_params = st.query_params

    # Step 1: Handle login
    if "access_token" not in st.session_state:
        if "code" not in query_params:
            auth_url = (
                f"https://accounts.spotify.com/authorize"
                f"?client_id={CLIENT_ID}"
                f"&response_type=code"
                f"&redirect_uri={REDIRECT_URI}"
                f"&scope=playlist-modify-private playlist-modify-public"
            )
            st.markdown(
                f'<a href="{auth_url}" target="_blank">🔑 Login with Spotify</a>',
                unsafe_allow_html=True
            )
            return

        code = query_params["code"]
        tokens = get_token(code)
        access_token = tokens.get("access_token")
        refresh_token = tokens.get("refresh_token")
        scope = tokens.get("scope") 

        if not access_token:
            st.error(f"Failed to get access token: {tokens}")
            return

        st.session_state["access_token"] = access_token
        st.session_state["refresh_token"] = refresh_token
        st.session_state["scope"] = scope

    # Step 2: Ensure token is valid (refresh if needed)
    access_token = st.session_state["access_token"]
    test_response = requests.get("https://api.spotify.com/v1/me", headers={"Authorization": f"Bearer {access_token}"})
    if test_response.status_code == 401:
        refresh_token = st.session_state.get("refresh_token")
        if refresh_token:
            new_tokens = refresh_access_token(refresh_token)
            if "access_token" in new_tokens:
                st.session_state["access_token"] = new_tokens["access_token"]
                access_token = new_tokens["access_token"]

    # Step 3: Playlist creation button
    if st.button("🎶 Create Playlist from Unique Songs"):
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Get user info
        progress_text.text("📀 Step 1/3: Fetching user info...")
        user_info = get_spotify_user_info(access_token)
        st.write(user_info)
        progress_bar.progress(20)
        if not user_info:
            return

        # Create playlist
        # 🎤 Ask for playlist name in a modal if filters changed
        playlist_name = "United Kingdom Dashboard Playlist"
        if is_any_filter_different:
            with st.form("playlist_name_form", clear_on_submit=True):
                st.write("📝 Please enter a name for your playlist:")
                playlist_name = "United Kingdom Dashboard Playlist Filtered"
                # Show default name in the input field
                playlist_name_input = st.text_input(
                    "Playlist name",
                    value=playlist_name  # ✅ default pre-filled
                )
                
                submitted = st.form_submit_button("Confirm")
                if submitted:
                    playlist_name = playlist_name_input.strip() or playlist_name
                else:
                    st.warning("Playlist name required to continue.")
                
        playlist_description = build_playlist_description(
            start_date, end_date,
            selected_artists, collaboration_choice,
            selected_album_types, duration_range,
            selected_popularity, selected_genres,
            is_any_filter_different
        )
        
        progress_text.text("🎶 Step 2/3: Creating playlist...")
        playlist = create_spotify_playlist(access_token, name=playlist_name, description=playlist_description)
        progress_bar.progress(40)
        if "id" not in playlist:
            st.error(f"Failed to create playlist: {playlist}")
            return
        playlist_id = playlist["id"]
        upload_playlist_cover(playlist_id, "static/resized_Livestream_symbol.png", access_token)

        # Add tracks
        progress_text.text("⏳ Step 3/3: Adding tracks...")
        track_uris = []
        total = len(unique_songs)
        st.write(f"Scope: {st.session_state.get('scope')}")
        
        for i, (_, row) in enumerate(unique_songs.iterrows()):
            track_id = search_spotify_track(
                row["song"], row["artist"],
                {"Authorization": f"Bearer {access_token}"}
            )
            
            if track_id:  
                track_uris.append(f"spotify:track:{track_id[0]}")
        
            # update progress bar gradually
            percent_complete = 40 + int(60 * (i+1)/total)
            progress_bar.progress(percent_complete)
            progress_text.text(f"Searching track {i+1}/{total}...")
        
        # Debug check
        # Clean up track_uris before sending
        track_uris = [str(uri) for uri in track_uris if uri]
        st.write("Token user ID:", user_info["id"])
        st.write("Playlist owner ID:", playlist["owner"]["id"])
        st.write("Final track URIs:", track_uris[:5])
        st.write("Using token:", access_token)
        
        add_tracks_to_playlist(playlist_id, track_uris, access_token)

        progress_bar.progress(100)
        progress_text.text("✅ Playlist created successfully!")
        st.success("Playlist created successfully!")

        # Show owner info
        st.markdown(f"👤 Playlist owner: **{user_info.get('display_name', user_info.get('id'))}**")
        
        st.markdown(
            f"""
            <a href="https://open.spotify.com/playlist/{playlist_id}" target="_blank">
                <button style="background-color:#1DB954; border:none; color:white; padding:10px 20px; text-align:center; 
                    text-decoration:none; display:inline-block; font-size:16px; border-radius:20px; cursor:pointer;"> 🎧 Open Playlist in Spotify
                </button>
            </a>
            """,
            unsafe_allow_html=True
        )

def create_playlist_button(unique_songs, start_date, end_date, selected_artists, collaboration_choice, selected_album_types, duration_range,
                                selected_popularity, selected_genres, is_any_filter_different):
    if st.button("🎶 Create Playlist"):
        progress_bar = st.progress(0)
        progress_text = st.empty()

        # Step 1: Login + token exchange
        query_params = st.query_params
        if "access_token" not in st.session_state:
            if "code" not in query_params:
                auth_url = (
                    f"https://accounts.spotify.com/authorize"
                    f"?client_id={CLIENT_ID}"
                    f"&response_type=code"
                    f"&redirect_uri={REDIRECT_URI}"
                    f"&scope=playlist-modify-private playlist-modify-public"
                )
                st.markdown(f"[🔑 Login with Spotify]({auth_url})")
                return

            code = query_params["code"]
            tokens = get_token(code)
            if "access_token" not in tokens:
                st.error("Failed to get access token")
                return

            st.session_state["access_token"] = tokens["access_token"]
            st.session_state["refresh_token"] = tokens.get("refresh_token")

        access_token = st.session_state["access_token"]

        # Step 2: Get user info
        progress_text.text("📀 Fetching user info...")
        user_info = get_spotify_user_info(access_token)
        progress_bar.progress(20)
        if not user_info:
            return

        # Step 3: Create playlist
        # 🎤 Ask for playlist name in a modal if filters changed
        playlist_name = "United Kingdom Dashboard Playlist"
        if is_any_filter_different:
            with st.form("playlist_name_form", clear_on_submit=True):
                st.write("📝 Please enter a name for your playlist:")
                playlist_name = "United Kingdom Dashboard Playlist Filtered"
                # Show default name in the input field
                playlist_name_input = st.text_input(
                    "Playlist name",
                    value=playlist_name  # ✅ default pre-filled
                )
                
                submitted = st.form_submit_button("Confirm")
                if submitted:
                    playlist_name = playlist_name_input.strip() or playlist_name
                else:
                    st.warning("Playlist name required to continue.")
                    return
                
        playlist_description = build_playlist_description(
            start_date, end_date,
            selected_artists, collaboration_choice,
            selected_album_types, duration_range,
            selected_popularity, selected_genres,
            is_any_filter_different
        )
        
        progress_text.text("🎶 Creating playlist...")
        playlist = create_spotify_playlist(access_token, playlist_name, playlist_description)
        progress_bar.progress(40)
        if "id" not in playlist:
            st.error("Failed to create playlist")
            return
        playlist_id = playlist["id"]

        # Step 4: Collect track URIs
        progress_text.text("⏳ Searching tracks...")
        track_uris = []
        total = len(unique_songs)
        for i, (_, row) in enumerate(unique_songs.iterrows()):
            track_id = search_spotify_track(
                row["song"], row["artist"],
                {"Authorization": f"Bearer {access_token}"}
            )
            if track_id:
                track_uris.append(f"spotify:track:{track_id[0]}")

            percent_complete = 40 + int(60 * (i+1)/total)
            progress_bar.progress(percent_complete)
            progress_text.text(f"Searching track {i+1}/{total}...")

        # Step 5: Add tracks
        add_tracks_to_playlist(playlist_id, track_uris, access_token)

        progress_bar.progress(100)
        progress_text.text("✅ Playlist created successfully!")
        st.success("Playlist created successfully!")

        # Show owner info + link
        st.markdown(f"👤 Playlist owner: **{user_info.get('id')}**")
        st.markdown(
            f"""
            <a href="https://open.spotify.com/playlist/{playlist_id}" target="_blank">
                <button style="background-color:#1DB954;border:none;color:white;
                                padding:10px 20px;border-radius:20px;cursor:pointer;">
                    🎧 Open Playlist in Spotify
                </button>
            </a>
            """,
            unsafe_allow_html=True
        )
